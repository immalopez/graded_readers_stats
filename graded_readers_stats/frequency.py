##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################
import math

from pandas import Int64Index

from graded_readers_stats.config import log
from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def count_vocab_in_texts_grouped_by_level(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str,
        is_context: bool = False
) -> (str, Series):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        in_column_locations = column + ctx_name + LOCATIONS
        out_column_counts = column + ctx_name + COUNTS + ' ' + group_name

        counts = phrases.apply(
            lambda x: count_phrase_occurrences_v1(
                x[in_column_locations],
                sentences_by_groups.get_group(group_name).index
            ),
            axis=1
        )
        results.append((out_column_counts, counts))
    return results


def count_phrase_occurrences_v1(
        locations: [[(int, (int, int))]],  # list of docs of sentences
        group_indices: Int64Index
) -> int:
    # We count sentences since we detect the first occurrence of the vocab
    # meaning vocab appear at most once in a sentence.
    # NOTE: We are using '_' to indicate that we don't need the variable.
    return sum(1
               # iterate over pandas' series to recover index
               for doc_index in group_indices
               # we count sentences since we detect only
               # the first occurrence of the vocab in a sentence.
               # iterate over sentences in document
               for _ in locations[doc_index])


def frequency(
        locations: [[(int, (int, int))]],  # list of docs of sentences
) -> float:
    docs = locations
    return sum(1 for _ in docs)


def total_count_in_texts_grouped_by_level(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str,
        is_context: bool = False
):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        out_column_counts = column + ctx_name + TOTAL_COUNTS + ' ' + group_name

        from pandas.core.common import flatten

        lemmas = sentences_by_groups.get_group(group_name)[COL_LEMMA]
        total_count = sum(1 for _ in flatten(lemmas))
        totals = vocabs.apply(lambda x: total_count, axis=1)
        results.append((out_column_counts, totals))
    return results


@DeprecationWarning
def total_counts_for_vocab(
        locations: [[(int, (int, int))]],
        texts: Series  # (2, [['Text', 'items']], ...)
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
        column: str,
        is_context: bool = False
) -> (str, Series):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        column_count = column + ctx_name + COUNTS + ' ' + group_name
        column_total_count = column + ctx_name + TOTAL_COUNTS + ' ' + group_name
        column_frequency = column + ctx_name + FREQUENCY + ' ' + group_name

        freqs = vocabs.apply(
            lambda x: x[column_count] / x[column_total_count]
            if x[column_total_count] > 0 else 0,
            axis=1
        )
        results.append((column_frequency, freqs))
    return results


def tfidfs_for_groups(locs, doc_groups, column_id) -> pd.DataFrame:
    # Input: locations, documents
    # Output: a dataframe with TF-IDF columns ready to be merged

    # For each term in the vocabulary:

    # 0. Normalize
    # Convert join all texts and treat sentences as documents.
    # Requires averaging at the end to achieve a single value per term.

    # 1. Calculate the term frequency for each document
    # where a document is a sentence in the corpus.

    # 2. Calculate the inverse document frequency for each term
    # by dividing the number of documents (sents) in the group by level
    # by the number of documents where the term appears.

    # 3. Calculate the TF-IDF for each document.

    # 4. Average the TF-IDF for each document
    # and connect it with the term.

    result = {}
    # For each group:
    for group_name, group_df in doc_groups:

        # For each term in the vocabulary:
        term_result = []
        for term_locs in locs:

            # For each document in the group:
            tfidfs_group = []
            for doc_idx in group_df.index:
                doc_sents = group_df['Lemma'].loc[doc_idx]
                doc_matches = term_locs[doc_idx]

                sents_count = len(doc_sents)
                sents_matched = len(doc_matches)

                if sents_matched > 0:
                    idf_doc = math.log10(sents_count / sents_matched)

                    tfs_sents = []
                    for doc_match in doc_matches:
                        term_start = doc_match[1][0]
                        term_end = doc_match[1][1]
                        term_word_count = term_end - term_start
                        sent_idx = doc_match[0]
                        sent_words = doc_sents[sent_idx]
                        sent_word_count = len(sent_words)
                        # count multi-word terms as 1
                        # by removing its length and adding 1 instead
                        tf = 1 / (sent_word_count - term_word_count + 1)
                        tfs_sents.append(tf)

                    tfidfs_sents = map(lambda x: x * idf_doc, tfs_sents)
                    tfidf_doc_avg = sum(tfidfs_sents) / sents_count
                    tfidfs_group.append(tfidf_doc_avg)

            # TF-IDF for current term
            tfidf_group_avg = sum(tfidfs_group) / len(group_df)
            term_result.append(tfidf_group_avg)

        # save the averaged series of vocab terms tfidfs
        result[TFIDF + '_' + column_id + '_' + group_name] = term_result

    # a dataframe with columns by group ready to be merged
    # into the main dataframe
    return pd.DataFrame(result)


def tfidfs(terms_locs, docs):
    """
    term_locs
    ---
    A matrix with rows corresponding to terms and columns corresponding
    to docs. If there are 2 terms and 3 docs it would be a 2x3 matrix
    filled with indices of sentences where a term is found.
    Shape: [
        [[(sent_index, (term_start_index, term_end_index))]],
        [[...]],
        ...,
    ]

    docs
    ---
    docs is a list, where each
        doc is a list of sents, where each
            sent is a list of words.
    [
        [['This', 'is', 'a', 'sentence', '.'], ['And', 'this', 'as', 'well']],
        [['Another', 'sentence']]
    ]

    returns
    ---
    a list of TFIDF e.g. [0.0458, 0.0058, ...] where all the locations for a
    term are reduced to a single average TFIDF.

    """
    # For each term in the vocabulary:
    result_tfidfs_per_term = []
    for term_locs in terms_locs:

        # For each document in the group:
        tfidfs_group = []
        for doc_idx, doc_sents in enumerate(docs):
            doc_matches = term_locs[doc_idx]

            sents_count = len(doc_sents)
            sents_matched = len(doc_matches)

            if sents_matched > 0:
                idf_doc = math.log10(sents_count / sents_matched)

                tfs_sents = []
                for doc_match in doc_matches:
                    term_start = doc_match[1][0]
                    term_end = doc_match[1][1]
                    term_word_count = term_end - term_start
                    sent_idx = doc_match[0]
                    sent_words = doc_sents[sent_idx]
                    # count multi-word terms as 1
                    # by removing its length and adding 1 instead
                    sent_word_count = (len(sent_words) - term_word_count + 1)
                    # use frequency 1 since we match 1 term per sentence max
                    tf = 1 / sent_word_count
                    tfs_sents.append(tf)

                tfidfs_sents = map(lambda tf: tf * idf_doc, tfs_sents)
                tfidfs_doc_sum = sum(tfidfs_sents)
                tfidf_doc_avg = tfidfs_doc_sum / sents_count
                tfidfs_group.append(tfidf_doc_avg)

        # TF-IDF for current term
        tfidf_group_avg = sum(tfidfs_group) / len(docs)
        result_tfidfs_per_term.append(tfidf_group_avg)

    return result_tfidfs_per_term


def freqs_by_term(terms_locs, words_total):
    terms_counts = [sum(1 for sents in term_locs for _ in sents)
                    for term_locs in terms_locs]
    log('\tfreqs_by_term() -> terms_counts:', terms_counts)
    return [count / words_total for count in terms_counts]
