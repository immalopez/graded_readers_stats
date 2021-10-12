###############################################################################
#                              CORPUS STATISTICS                              #
###############################################################################

from nltk.probability import FreqDist
import pandas as pd
import numpy as np

readers = pd.Series(data=[
    [
        ['cara', 'a'],
        ['cara', 'a', 'b'],
        # ['¿', 'ya', 'poder', 'empezar', '?'],
        # ['bueno', ',', 'este', 'ser', 'difícil', '.']
    ],
    [
        # ['peter', 'pan'],
        # ['system', 'out', 'print', 'line', '?']
    ]
])

vocab = pd.Series(data=[
    [['cara', 'a']],
    # [['¿', 'cómo', 'estar', '?']],
    # [['¿', 'qué', 'hora', 'ser', '?']]
])

def array_contains_array(arr1, arr2):
    if len(arr1) > len(arr2):
        return False

    for a1 in arr1:
        for a2 in arr2:
            if a1 != a2:
                return False

    return True

# print(array_contains_array([2], [2, 4]))


def isSubArray(A, B):
    n = len(A)
    m = len(B)
    # Two pointers to traverse the arrays
    i = 0
    j = 0

    # Traverse both arrays simultaneously
    while i < n and j < m:

        # If element matches
        # increment both pointers
        if A[i] == B[j]:

            i += 1
            j += 1

            # If array B is completely
            # traversed
            if j == m:
                return True

        # If not,
        # increment i and reset j
        else:
            i = i - j + 1
            j = 0

    return False


print(isSubArray([3, 2, 4], [4]))


def compute_freq_dist(text, vocabulary):
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        # print(vocab_item)

        for text_items in text:
            for text_item in text_items:
                print('comparing: ' + str(vocab_item) + ' in ' + str(text_item))
                print(isSubArray(text_item, vocab_item))
                # print(array_contains_array(vocab_item, text_item))
                # print(np.array_equal(vocab_item, text_item))
                # print(vocab_item.equal(text_item))


compute_freq_dist(readers, vocab)











# return FreqDist(lexical_item
#                 for lexical_item in text if lexical_item in vocabulary)
# print(1 in [1, 2])
