##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################
from pandas import Int64Index

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def count_vocab_in_texts_grouped_by_level(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str
) -> None:
    for group_name in sentences_by_groups.groups:
        in_column_locations = column + ' ' + LOCATIONS
        out_column_counts = column + ' ' + COUNTS + ' ' + group_name
        phrases[out_column_counts] = phrases.apply(
            lambda x: count_phrase_occurrences_v1(
                x[in_column_locations],
                sentences_by_groups.get_group(group_name).index
            ),
            axis=1
        )


def count_phrase_occurrences_v1(
        locations: [[(int, (int, int))]],  # list of docs of sentences
        group_indices: Int64Index
) -> int:
    # We count sentences since we detect the first occurrence of the vocab
    # meaning vocabs appear at most once in a sentence.
    # NOTE: We are using '_' to indicate that we don't need the variable.
    return sum(1
               # iterate over pandas' series to recover index
               for doc_index in group_indices
               # we count sentences since we detect only
               # the first occurrence of the vocab in a sentence.
               # iterate over sentences in document
               for _ in locations[doc_index])


def total_count_in_texts_grouped_by_level(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str
):
    for group_name in sentences_by_groups.groups:
        in_column_locations = column + ' ' + LOCATIONS
        out_column_counts = column + ' ' + TOTAL_COUNTS + ' ' + group_name
        vocabs[out_column_counts] = vocabs.apply(
            lambda x: total_counts_for_vocab(
                x[in_column_locations],
                sentences_by_groups.get_group(group_name)[COL_LEMMA]
            ),
            axis=1
        )


def total_counts_for_vocab(
        locations: [[(int, (int, int))]],
        texts: Series  # (2, [['Text', items']], ...)
) -> int:

    # Print the whole document (all sentences in doc)
    # print(
    #     [[w for sent in sents for w in sent]
    #      for doc_index, sents in texts.items()
    #      if len(locations[doc_index]) > 0]
    # )

    # Count number of words in a sentence.
    # Shape: [[1, 3, 1]], thus we need to unwrap outer list.
    # When there are no occurrences, we deal with an empty list [].
    total = [[len(sent) for sent in doc_sents]

             # iterate over pandas' series to recover index
             for doc_index, doc_sents in texts.items()

             # only docs with occurrences
             if len(locations[doc_index]) > 0]

    # sum word counts for all documents
    return sum(total[0]) if len(total) > 0 else 0


def frequency_in_texts_grouped_by_level(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str
) -> None:
    for group_name in sentences_by_groups.groups:
        column_count = column + ' ' + COUNTS + ' ' + group_name
        column_total_count = column + ' ' + TOTAL_COUNTS + ' ' + group_name
        column_frequency = column + ' ' + FREQUENCY + ' ' + group_name
        vocabs[column_frequency] = vocabs.apply(
            lambda x: x[column_count] / x[column_total_count]
            if x[column_total_count] > 0 else 0,
            axis=1
        )
