##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from graded_readers_stats._typing import *


def count_phrases_in_sentences_by_groups(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        phrases[output_column_name] = phrases.apply(
            lambda x: count_phrase_in_sentences(
                x['Lemma'][0],
                sentences_by_groups.get_group(group_name)['Lemma']
            ),
            axis=1
        )


def count_phrase_in_sentences(phrase: [str], texts: Series) -> int:
    count = 0
    for sents in texts:
        for sent in sents:
            if get_range_of_phrase_in_sentence(phrase, sent):
                count += 1
    return count


def count_context_in_sentences_by_groups(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        phrases[output_column_name] = phrases.apply(
            lambda x: count_context_in_sentences(
                x['Reader_Context'],
                sentences_by_groups.get_group(group_name)['Lemma']
            ),
            axis=1
        )


def count_context_in_sentences(words: [str], texts: Series) -> int:
    count = 0
    for word in words:
        for sents in texts:
            for sent in sents:
                if get_range_of_phrase_in_sentence([word], sent):
                    count += 1
    return count


def collect_context_for_phrases_in_texts(
        phrases: DataFrame,
        texts: DataFrame,
        column_prefix: str
) -> None:
    phrases[column_prefix + 'Context'] = phrases.apply(
        lambda x: collect_context_for_phrase_in_texts(
            x['Lemma'][0],
            texts['Lemma']
        ),
        axis=1
    )


def collect_context_for_phrase_in_texts(
        phrase: [str],
        texts: Series,
        window: int = 3
) -> [str]:

    context = []
    for sentences in texts:
        for sent in sentences:
            text_range = get_range_of_phrase_in_sentence(phrase, sent)
            if text_range:
                start, end = text_range[0], text_range[1]

                # limit indices to sentence bounds using `min` and `max`
                # to safely use with list slicing
                # since out-of-bounds slicing would return empty list ([])
                slice_before = slice(max(0, start - window), start)
                slice_after = slice(end, min(end + window, len(sent)))

                words = sent[slice_before] + sent[slice_after]
                context.extend(words)

    return context


def get_range_of_phrase_in_sentence(phrase: [str], sentence: [str]):
    """Returns a tuple(start, end) where start is the first index of
    vocabulary in text and end is the end index of the vocabulary in text if
    the lexical items contained in a vocabulary list are to be found in the
    sentence lists of a given text, and None otherwise."""
    sent_index = 0
    sent_len = len(sentence)
    phrase_index = 0
    phrase_len = len(phrase)
    while sent_index < sent_len and phrase_index < phrase_len:
        if str(sentence[sent_index]).lower() == str(phrase[phrase_index]).lower():
            sent_index += 1
            phrase_index += 1
            if phrase_index == phrase_len:
                # adjust start to include vocab item(s)
                return sent_index - phrase_len, sent_index  # a tuple
        else:
            sent_index = sent_index - phrase_index + 1
            phrase_index = 0
    return None
