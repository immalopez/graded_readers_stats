##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################
from pandas import Int64Index

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
    # NOTE: Using '_' to indicate we don't care about the sentence index.
    return sum(1
               # iterate over pandas' series to recover index
               for doc_index in group_indices
               # we count sentences since we detect only
               # the first occurrence of the vocab in a sentence.
               # iterate over sentences in document
               for _ in locations[doc_index])


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


def total_counts_for_phrase(
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


def count_phrase_in_sentences_v0(vocab: [str], texts: Series) -> int:
    count = 0
    for sents in texts:
        for sent in sents:
            if first_occurrence_of_vocab_in_sentence(vocab, sent):
                count += 1
    return count


def count_vocab_context_in_sentences_by_groups(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column_prefix: str
) -> None:
    for group_name in sentences_by_groups.groups:
        output_column_name = column_prefix + group_name
        vocabs[output_column_name] = vocabs.apply(
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
                if first_occurrence_of_vocab_in_sentence([word], sent):
                    count += 1
    return count
