import math
import time

import numpy as np
from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_STANZA_DOC,
    COL_LEVEL,
)
from graded_readers_stats.context import (
    collect_context_words_by_terms,
    freqs_pipeline,
    tfidfs_pipeline, trees_pipeline, locate_ctx_terms_in_docs, count_pipeline,
    avg, collect_context_words_by_docs,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.frequency import (
    freqs_by_term,
    count_terms,
    count_doc_terms,
)
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    locate_terms_in_docs,
)
from graded_readers_stats.stats import get_msttr
from graded_readers_stats.tfidf import tfidfs, tfidfs_2
from graded_readers_stats.tree import (
    terms_tree_props_pipeline,
    texts_tree_props_pipeline,
)


def analyze(args):
    vocabulary_path = args.vocabulary_path
    corpus_path = args.corpus_path
    max_terms = args.max_terms
    max_docs = args.max_docs

    print()
    print('ANALYZE DOCS START')
    print('---')
    print('vocabulary_path = ', vocabulary_path)
    print('corpus_path = ', corpus_path)
    print('max_terms = ', max_terms)
    print('max_docs = ', max_docs)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

    with Timer(name='Load data', text=timer_text):
        terms_df = read_pandas_csv(vocabulary_path)
        texts_df = read_pandas_csv(corpus_path)
        if max_terms:
            terms_df = terms_df[:max_terms]
        if max_docs:
            texts_df = texts_df[:max_docs]

    with Timer(name='Preprocess', text=timer_text):
        texts_df = run(texts_df, text_analysis_pipeline)
        terms_df = run(terms_df, vocabulary_pipeline)
        texts = texts_df[COL_LEMMA]
        storage = {
            'stanza': texts_df[COL_STANZA_DOC],
            'tree': {}
        }
        texts_df = texts_df.drop(columns=[
            COL_STANZA_DOC,
            "Publisher",
            "Text file",
            "Type",
        ])
        terms_df = terms_df.drop(columns=[
            COL_STANZA_DOC,
            "Topic",
            "Subtopic",
        ])

##############################################################################
#                                 Terms                                      #
##############################################################################

    with Timer(name='Locate terms', text=timer_text):
        terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
        terms_locs = locate_terms_in_docs(terms, texts)

    with Timer(name='Frequency', text=timer_text):
        docs_locs = list(zip(*terms_locs))
        levels = terms_df[COL_LEVEL].unique()
        for level in levels:
            term_indices = terms_df \
                .groupby(COL_LEVEL) \
                .get_group(level) \
                .index \
                .to_list()
            texts_df[f'Count {level}'] = count_doc_terms(
                docs_locs,
                term_indices
            )
        texts_df["Count"] = count_doc_terms(
            docs_locs,
            term_indices=range(0, len(terms_df))  # all terms
        )
        texts_df["Total"] = texts_df["Lemma"]\
            .apply(lambda x: sum(1 for _ in flatten(x)))
        for level in levels:
            texts_df[f"Freq {level}"] \
                = texts_df[f"Count {level}"] / texts_df["Total"]
        texts_df[f"Freq"] \
            = texts_df[f"Count"] / texts_df["Total"]

    with Timer(name='TFIDF', text=timer_text):
        docs_matches = [
            [1 if len(doc) else 0 for doc in doc_locs]
            for doc_locs in docs_locs
        ]
        # [1, 1, 0, 1, 1]
        # [0, 0, 1, 1, 0]

        docs_match_counts = [
            sum(doc_matches)
            for doc_matches in docs_matches
        ]
        # [4, 2]

        term_matches = [sum(column) for column in zip(*docs_matches)]
        # [1, 1, 1, 2, 1]

        doc_count = len(docs_locs)
        # 2

        term_idfs = [
            math.log10(doc_count / matched) if matched > 0 else 0
            for matched in term_matches
        ]
        # [0.3010, 0.3010, 0.3010, 0.0000, 0.3010]

        docs_idfs = [
            [term_idfs[idx] if bit else 0 for idx, bit in enumerate(doc_mask)]
            for doc_mask in docs_matches
        ]
        # [
        #   [0.3010, 0.3010, 0, 0.0, 0.3010],
        #   [0, 0, 0.3010, 0.0, 0]
        # ]

        doc_avg_idfs = [
            sum(doc_idfs) / docs_match_counts[doc_idx]
            if docs_match_counts[doc_idx] > 0 else 0
            for doc_idx, doc_idfs in enumerate(docs_idfs)
        ]
        # [0.2257, 0.1505]

        texts_df["IDF"] = doc_avg_idfs
        texts_df['TFIDF'] = texts_df["Freq"] * texts_df["IDF"]
        # 0.03099
        # 0.01881

        texts_df = texts_df.drop(columns="IDF")

    with Timer(name='Tree', text=timer_text):
        texts_df["Tree"] = texts_tree_props_pipeline(storage, docs_locs)

##############################################################################
#                                Contexts                                    #
##############################################################################

    with Timer(name='Context collect', text=timer_text):
        ctx_words_by_terms = collect_context_words_by_terms(
            terms_locs,
            texts,
            window=3
        )
        ctx_words_by_docs = collect_context_words_by_docs(
            docs_locs,
            texts,
            window=3
        )
        texts_df['Context words'] = ctx_words_by_docs

    with Timer(name='Context locate terms', text=timer_text):
        ctxs_locs_by_terms = locate_ctx_terms_in_docs(ctx_words_by_terms, texts)
        terms_count = len(terms_df)
        ctxs_locs_by_docs = [[] for _ in range(doc_count)]
        for term_idx, term in enumerate(ctxs_locs_by_terms):
            for ctxterm_idx, ctxterm in enumerate(term):
                for doc_idx, doc_locs in enumerate(ctxterm):
                    if len(doc_locs):
                        while len(ctxs_locs_by_docs[doc_idx]) <= ctxterm_idx:
                            ctxs_locs_by_docs[doc_idx].append(
                                [[] for _ in range(terms_count)]
                            )
                        ctxs_locs_by_docs[doc_idx][ctxterm_idx][term_idx] = doc_locs

    with Timer(name='Context frequency', text=timer_text):
        print()
        ctx_counts = [  # list of docs
            [   # list of ctx term counts
                sum(1 for term in ctxterm for loc in term)
                for ctxterm in doc
            ]
            for doc in ctxs_locs_by_docs
        ]
        texts_df['Context count per word'] = ctx_counts
        texts_df['Context count'] = list(map(avg, ctx_counts))
        texts_df['Context total'] = texts_df['Total']
        texts_df['Context frequency'] = \
            texts_df['Context count'] / texts_df['Context total']
#
#     with Timer(name='Context TFIDF', text=timer_text):
#         terms_df['Context TFIDF'] = list(tfidfs_pipeline(texts)(ctxs_locs))
#
#     with Timer(name='Context Tree', text=timer_text):
#         terms_df['Context tree'] = list(trees_pipeline(storage)(ctxs_locs))
#
# ##############################################################################
# #                                  Others                                    #
# ##############################################################################
#
#     with Timer(name='MSTTR', text=timer_text):
#         joined_text = ' '.join(texts_df['Raw text'])
#         print(f'{get_msttr(joined_text)}')
#
#     with Timer(name='Export CSV', text=timer_text):
#         terms_df.to_csv(f'./output/{level}.csv', index=False)
#
#     print()
#     utils.duration(start_main, 'Total time')
#     print('')
#     print('ANALYZE DOCS END')
