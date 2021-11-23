from typing import Optional, Tuple


def calculatestar(args):
    return calculate(*args)


def calculate(func, args):
    return func(*args)


def first_occurrence_of_vocab_in_sentence(
        vocab: [str],
        sentence: [str]
) -> Optional[Tuple[int, int]]:
    """Returns a tuple(start, end) where start is the first index of
    vocabulary in text and end is the end index of the vocabulary in text if
    the lexical items contained in a vocabulary list are to be found in the
    sentence lists of a given text, and None otherwise."""
    sent_index = 0
    sent_len = len(sentence)
    phrase_index = 0
    phrase_len = len(vocab)
    while sent_index < sent_len and phrase_index < phrase_len:
        if str(sentence[sent_index]).lower() == str(vocab[phrase_index]).lower():
            sent_index += 1
            phrase_index += 1
            if phrase_index == phrase_len:
                # adjust start to include vocabs item(s)
                return sent_index - phrase_len, sent_index  # a tuple
        else:
            sent_index = sent_index - phrase_index + 1
            phrase_index = 0
    return None
