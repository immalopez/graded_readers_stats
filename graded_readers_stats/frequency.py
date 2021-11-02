##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from graded_readers_stats._typing import *
from graded_readers_stats.utils import *
from graded_readers_stats.constants import *


def count_vocab_in_sentences_by_groups(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        phrases[output_column_name] = phrases.apply(
            lambda x: count_phrase_in_sentences(
                x[COL_LEMMA][0],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
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


def count_vocab_context_in_sentences_by_groups(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        phrases[output_column_name] = phrases.apply(
            lambda x: count_context_in_sentences(
                x[PREFIX_READER + SUFFIX_CONTEXT],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
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


