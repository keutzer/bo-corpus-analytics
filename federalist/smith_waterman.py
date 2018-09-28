import numpy as np

A = "the existence of the union".split()
B = "the existence of the state".split()


def score(x, y, match, mismatch):
    return match if x == y else mismatch


def smith_waterman(A, B, match=1, mismatch=-1, gap_open=10, gap_ext=0.5):
    n = len(A)
    m = len(B)

    H = np.zeros((n + 1, m + 1))
    # TODO: change to default dict and to visit all paths
    S = {}
    max_i = -1
    max_j = -1
    max_val = -float('inf')

    W = np.array([gap_open + gap_ext * (k - 1) for k in range(1, max(n, m) + 1)])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opt1 = H[i - 1, j - 1] + score(A[i - 1], B[j - 1], match, mismatch)
            opt2 = np.max(H[:i, j] - np.flip(W[:i], 0))
            opt3 = np.max(H[i, :j] - np.flip(W[:j], 0))
            opt4 = 0
            entry = max(opt1, opt2, opt3, opt4)

            # Track source paths for traceback
            if entry == opt1:
                S[(i, j)] = (i - 1, j - 1)
            elif entry == opt2:
                S[(i, j)] = (i - 1, j)
            elif entry == opt3:
                S[(i, j)] = (i, j - 1)

            # TODO: complete code for all paths traceback
            # if entry > max_val:
            #     max_val = entry
            #     max_pos = [(i, j)]
            # elif entry == max_val:
            #     max_pos.append((i, j))

            if entry > max_val:
                max_val = entry
                max_pos = (i, j)

            H[i, j] = entry
 
    # aligns = []

    # for p in max_pos:
    #     matched_A = []
    #     matched_B = []

    #     take_A = True
    #     take_B = True

    #     i, j = p

    #     while True:
    #         if take_A:
    #             matched_A.append(A[i])
    #         else:
    #             matched_A.append('_')

    #         if take_B:
    #             matched_B.append(B[j])
    #         else:
    #             matched_B.append('_')

    #         if H[i, j] == 0:
    #             break



    #     aligns.append([reverse(matched_A), reverse(matched_B)])

    # return aligns

    matched_A = []
    matched_B = []

    take_A = True
    take_B = True

    i, j = max_pos

    while True:
        if H[i, j] == 0:
            break

        new_i, new_j = S[(i, j)]

        take_A = (i != new_i)
        take_B = (j != new_j)
        
        if take_A:
            matched_A.append(A[i - 1])
        else:
            matched_A.append('_')

        if take_B:
            matched_B.append(B[j - 1])
        else:
            matched_B.append('_')

        i = new_i
        j = new_j

    return matched_A[::-1], matched_B[::-1]

