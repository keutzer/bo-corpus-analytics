# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn

import data
import model

import mlflow

parser = argparse.ArgumentParser(description='PyTorch Tibetan Word Segmenter RNN/LSTM Model')
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35, 
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--mlflow', type=bool, default=True, help='Log with mlflow')
args = parser.parse_args()

if args.mlflow:
    mlflow.log_param("model", args.model)
    mlflow.log_param("emsize", args.emsize)
    mlflow.log_param("nhid", args.nhid)
    mlflow.log_param("nlayers", args.nlayers)
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("clip", args.clip)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("dropout", args.dropout)
    mlflow.log_param("seed", args.seed)   


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, args.bptt)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 2, we'd get
#    Data        Targets      Seq Len

#   batch 1:
# ┌ a e h j ┐  ┌ 1 1 0 0 ┐  [4, 3, 2, 4]
# │ b f i k │  │ 0 0 1 1 │
# │ c g _ l │  │ 1 0 _ 1 │
# │ d _ _ m │  │ 1 _ _ 1 │
#   batch 2:
# │ ...     │  │ ...     │      ...

# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'e' on 'd' can not be learned, but allows more efficient
# batch processing.

# Additionally, the above cartoon is not completely accurate- 
# In reality the sample "efg" would be element 0 in batch 2, right after "abcd"

def batchify(data, labels, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = (data.size(0) * data.size(1)) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.view(-1).narrow(0, 0, nbatch * bsz)
    labels = labels.view(-1).narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    labels = labels.view(bsz, -1).t().contiguous()

    return data.to(device), labels.to(device)

eval_batch_size = 10
train_data, train_labels = batchify(corpus.train_data, corpus.train_labels, args.batch_size)
val_data, val_labels = batchify(corpus.valid_data, corpus.valid_labels, args.batch_size)
test_data, test_labels = batchify(corpus.test_data, corpus.test_labels, args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print("Dictionary size: {}".format(ntokens))
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)

###############################################################################
# Training code
###############################################################################


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ 1 1 1 0 ┐
# └ b h n t ┘ └ 0 0 1 0 ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(data, labels, seq_lens, i):
    data_batch = data[i:i+args.bptt]
    label_batch = labels[i:i+args.bptt]
    seq_len_batch = seq_lens[i:i+args.bptt]
    return data_batch, target_batch, seq_len_batch


def evaluate(data, labels, seq_lens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    total_preds = 0
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, args.bptt):
            hidden = model.init_hidden(args.batch_size)
            d, l, s = get_batch(data, labels, seq_lens, i)
            output, _ = model(d, hidden)
            mask = (data >= 0).float()
            loss, npreds = model.loss(output, labels, mask)
            output_flat = output.view(-1, ntokens)
            loss_tensor = criterion(output_flat, l).view(output.size(0), output.size(1), -1)
            total_loss += npreds * loss.item()
            total_preds += npreds
    return total_loss / total_preds


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, labels = get_batch(train_data, train_labels, corpus.train_seq_lens, i)
        hidden = model.init_hidden(args.batch_size)
        model.zero_grad()
        output, _ = model(data, hidden)
        mask = (data >= 0).float()
        loss, _ = model.loss(output, labels, mask)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, val_labels, corpus.val_seq_lens)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
