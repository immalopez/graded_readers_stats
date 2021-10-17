###############################################################################
#                              CORPUS STATISTICS                              #
###############################################################################

from nltk.probability import FreqDist
import pandas as pd
# using . to import code from same folder
from .utils import vocab_item_to_key

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


def get_vocab_in_texts_freq(vocabulary: pd.Series,
                            texts: pd.Series,
                            levels: pd.Series,
                            level_names: [str]):
    counts_per_level = {}
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        vocab_item_key = vocab_item_to_key(vocab_item)
        counts_per_level[vocab_item_key] = {
            level_names[0]: [0, 0],
            level_names[1]: [0, 0],
            level_names[2]: [0, 0]
        }
        for text_items, level in zip(texts, levels):
            for text_item in text_items:
                counts_per_level[vocab_item_key][level][1] += 1
                if check_if_vocab_in_text(text_item, vocab_item):
                    counts_per_level[vocab_item_key][level][0] += 1
    return counts_per_level


def get_vocab_in_text(text, vocabulary):
    """Returns a boolean list whose values depend on whether each item in the
    vocabulary can be found in any of the texts or not."""
    vocab_in_text = []
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        item_found: bool = False
        for text_items in text:
            if item_found:
                break
            for text_item in text_items:
                if check_if_vocab_in_text(text_item, vocab_item):
                    item_found = True
                    vocab_in_text.append(True)
                    break
        if not item_found:
            vocab_in_text.append(False)
    return vocab_in_text


def check_if_vocab_in_text(text, vocabulary):
    """Returns True if the lexical items contained in a vocabulary list are
    to be found in the sentence lists of a given text, and False otherwise."""
    t_pointer = 0
    v_pointer = 0
    while t_pointer < len(text) and v_pointer < len(vocabulary):
        if str(text[t_pointer]).lower() == str(vocabulary[v_pointer]).lower():
            t_pointer += 1
            v_pointer += 1
            if v_pointer == len(vocabulary):
                return True
        else:
            t_pointer = t_pointer - v_pointer + 1
            v_pointer = 0
    return False


def compute_vocab_freq_dist(vocabulary, vocabulary_in_text, text):
    return FreqDist(vocabulary[vocabulary_in_text].lower()
                    for vocabulary[vocabulary_in_text] in text)


print(get_vocab_in_text(readers, vocab))

# return FreqDist(lexical_item
#                 for lexical_item in text if lexical_item in vocabulary)
# print(1 in [1, 2])
