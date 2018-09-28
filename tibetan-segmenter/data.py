import os
import torch

tsheg = '་'
neg_ones = [0 for _ in range(100)]

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_line(self, line):
        return [self.add_word(w) for w in line]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, max_len):
        self.dictionary = Dictionary()
        r = self.tokenize(os.path.join(path, 'train_text.txt', 'train_target.txt'))
        self.train_data, self.train_label, self.train_seq_len = r
        r = self.tokenize(os.path.join(path, 'valid_text.txt', 'valid_target.txt'))
        self.valid_data, self.valid_label, self.valid_seq_len = r
        r = self.tokenize(os.path.join(path, 'test_text.txt', 'test_target.txt'))
        self.test_data, self.test_label, self.test_seq_len = r
        self.max_len = max_len

    # Returns padded data and target 2D arrays
    # With max_len = 4 we get something that looks like:

    #    Data        Targets    Seq Len
    # ┌ a b c d ┐  ┌ 1 0 1 1 ┐     4
    # │ e f g _ │  │ 1 0 0 _ │     3
    # │ h i _ _ │  │ 0 1 _ _ │     2
    # │ j k l m │  │ 0 1 1 1 │     4
    # │ n _ _ _ │  │ 1 _ _ _ │     1
    # └ o p q r ┘  └ 0 0 1 0 ┘     4

    def tokenize(self, text_path, target_path):
        """Tokenizes a text file."""
        assert os.path.exists(text_path)
        assert os.path.exists(target_path)
        # Add words to the dictionary
        ntokens = 0
        data = []
        labels = []
        seq_lens = []
        with open(target_path, 'r', encoding="utf-8-sig") as f_target:
            with open(text_path, 'r', encoding="utf-8-sig") as f_text:
                for l_target, l_text in zip(f_target, f_text):

                    # Process text
                    targets = l_target.split()
                    syllables = l_text.split(tsheg)
                    ntokens += len(targets)

                    assert len(words) == len(targets), "# syllables and # labels do not match"

                    for i in range(0, len(targets), self.max_len):
                        line_data = syllables[i:i+self.max_len]
                        line_labels = targets[i:i+self.max_len]
                        line_data = self.dictionary.add_line(line_data)
                        seq_len = len(line_data)
                        seq_lens.append(seq_len)

                        if seq_len < self.max_len:
                            line_data.extend(neg_ones[:self.max_len-seq_len])
                            line_labels.extend(neg_ones[:self.max_len-seq_len])
                        labels.append(line_labels)
                        data.append(line_data)

        return torch.LongTensor(data), torch.LongTensor(labels), seq_lens
