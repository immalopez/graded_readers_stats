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
        ['¿', 'ya', 'poder', 'empezar', '?']
        # ['bueno', ',', 'este', 'ser', 'difícil', '.']
    ],
    [
        ['peter', 'pan'],
        # ['system', 'out', 'print', 'line', '?']
    ]
])

vocab = pd.Series(data=[
    [['cara', 'a']],
    [['¿', 'cómo', 'estar', '?']],
    [['¿', 'ya', 'poder', 'empezar', '?']]
    # [['¿', 'qué', 'hora', 'ser', '?']]
])


def check_if_vocab_in_text(text, vocabulary):
    """Returns True if the lexical items contained in a vocabulary list are
    to be found in the sentence lists of a given text, and False otherwise."""
    t_length = len(text)
    v_length = len(vocabulary)
    pointer_t = 0
    pointer_v = 0
    while pointer_t < t_length and pointer_v < v_length:
        if str(text[pointer_t]).lower() == str(vocabulary[pointer_v]).lower():
            pointer_t += 1
            pointer_v += 1
            if pointer_v == v_length:
                return True
        else:
            pointer_t = pointer_t - pointer_v + 1
            pointer_v = 0
    return False


def get_vocab_in_text(text, vocabulary):
    """Returns a list of lists with all of the vocabulary lexical items that
    can be found in a given (set of) text(s)."""
    vocab_in_text = []
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        for text_items in text:
            for text_item in text_items:
                if (check_if_vocab_in_text(text_item, vocab_item) and
                        vocab_item not in vocab_in_text):
                    vocab_in_text.append(vocab_item)
    return vocab_in_text


def compute_vocab_freq_dist0(text, vocabulary):
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        for text_items in text:
            for text_item in text_items:
                print('comparing: ' + str(vocab_item) + ' in ' + str(
                    text_item))
                print(check_if_vocab_in_text(text_item, vocab_item))
                # print(array_contains_array(vocab_item, text_item))
                # print(np.array_equal(vocab_item, text_item))
                # print(vocab_item.equal(text_item))


print(get_vocab_in_text(readers, vocab))

# return FreqDist(lexical_item
#                 for lexical_item in text if lexical_item in vocabulary)
# print(1 in [1, 2])
