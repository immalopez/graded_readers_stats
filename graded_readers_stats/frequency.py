##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from graded_readers_stats._typing import *
from graded_readers_stats.utils import *
from graded_readers_stats.constants import *


def count_vocab_in_sentences_by_groups_v1(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str
) -> None:
    for group_name in sentences_by_groups.groups:
        in_column_locations = column + ' ' + LOCATIONS
        out_column_counts = column + ' ' + COUNTS + ' ' + group_name
        phrases[out_column_counts] = phrases.apply(
            lambda x: count_phrase_occurrences_v1(x[in_column_locations]),
            axis=1
        )


def total_counts_for_docs_with_vocab_occurrences(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str
):
    for group_name in sentences_by_groups.groups:
        in_column_locations = column + ' ' + LOCATIONS
        out_column_counts = column + ' ' + TOTAL_COUNTS + ' ' + group_name
        phrases[out_column_counts] = phrases.apply(
            lambda x: total_counts_for_phrase(
                x[in_column_locations],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
            ),
            axis=1
        )


def count_phrase_occurrences_v1(locations: [[(int, (int, int))]]) -> int:
    # We count sentences since we detect the first occurrence of the phrase
    # meaning phrases appear at most once in a sentence.
    # NOTE: Using '_' to indicate we don't care about the sentence index.
    return sum(1 for doc in locations for _ in doc)


def total_counts_for_phrase(
        locations: [[(int, (int, int))]],
        texts: Series
) -> int:

    # Print the whole document (all sentences in doc)
    # print(
    #     [[w for sent in sents for w in sent]
    #      for doc_index, sents in texts.items()
    #      if len(locations[doc_index]) > 0]
    # )

    # shape: [[1, 1, 1]], thus we need to unwrap outer list.
    # when there are no occurrences, we deal with an empty list [].
    total = [[1 for sent in sents for _ in sent]
             for doc_index, sents in texts.items()
             if len(locations[doc_index]) > 0]
    return sum(total[0]) if len(total) > 0 else 0


def count_vocab_in_sentences_by_groups_v0(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        phrases[output_column_name] = phrases.apply(
            lambda x: count_phrase_in_sentences_v0(
                x[COL_LEMMA][0],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
            ),
            axis=1
        )


def count_phrase_in_sentences_v0(phrase: [str], texts: Series) -> int:
    count = 0
    for sents in texts:
        for sent in sents:
            if first_occurrence_of_phrase_in_sentence(phrase, sent):
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
                x[READER + ' ' + CONTEXT],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
            ),
            axis=1
        )


def count_context_in_sentences(words: [str], texts: Series) -> int:
    count = 0
    for word in words:
        for sents in texts:
            for sent in sents:
                if first_occurrence_of_phrase_in_sentence([word], sent):
                    count += 1
    return count


