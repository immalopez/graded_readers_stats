###############################################################################
#                              CORPUS STATISTICS                              #
###############################################################################

import pandas as pd

# Using dot prefix (.) to import code from the same folder
from .utils import vocab_item_to_key


# ========= COMPUTE THE FREQUENCY DISTRIBUTION OF IN-TEXT VOCABULARY =========

def get_vocab_range_in_text(text, vocabulary):
    """Returns a tuple(start, end) where start is the first index in of
    vocabulary in text and end is the end index of the vocabulary in text if
    the lexical items contained in a vocabulary list are to be found in the
    sentence lists of a given text, and None otherwise. """
    t_pointer = 0
    v_pointer = 0
    len_text = len(text)
    len_vocab = len(vocabulary)
    while t_pointer < len_text and v_pointer < len_vocab:
        if str(text[t_pointer]).lower() == str(vocabulary[v_pointer]).lower():
            t_pointer += 1
            v_pointer += 1
            if v_pointer == len_vocab:
                # adjust start to include vocab item(s)
                return t_pointer - len_vocab, t_pointer  # a tuple
        else:
            t_pointer = t_pointer - v_pointer + 1
            v_pointer = 0
    return None


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
                if get_vocab_range_in_text(text_item, vocab_item):
                    item_found = True
                    vocab_in_text.append(True)
                    break
        if not item_found:
            vocab_in_text.append(False)
    return vocab_in_text


def get_vocab_in_texts_freq(vocabulary: pd.Series,
                            texts: pd.Series,
                            levels: pd.Series,
                            level_names: [str]):
    """Counts how many occurrences of each vocabulary item can be found in the
    texts of each language level."""
    counts_per_level = {}
    for vocab_items in vocabulary:
        vocab_item = vocab_items[0]
        vocab_item_key = vocab_item_to_key(vocab_item)

        counts_per_level[vocab_item_key] = {
            level_names[0]: [0, 0],  # Level 1: [Occurrences, Total]
            level_names[1]: [0, 0],  # Level 2: [Occurrences, Total]
            level_names[2]: [0, 0]   # Level 3: [Occurrences, Total]
        }
        for text_items, level in zip(texts, levels):
            for text_item in text_items:

                # Increment total count
                counts_per_level[vocab_item_key][level][1] += 1

                # Increment occurrences
                if get_vocab_range_in_text(text_item, vocab_item):
                    counts_per_level[vocab_item_key][level][0] += 1

    return counts_per_level
