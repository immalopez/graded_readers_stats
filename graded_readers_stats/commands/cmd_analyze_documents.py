import time

from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
)
from graded_readers_stats.context import (
    avg,
    collect_context_words_by_docs,
    collect_context_words_by_terms,
    locate_ctx_terms_in_docs,
    tfidfs_pipeline, transpose_ctx_terms_to_docs_locations, to_list_of_dicts,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.frequency import (
    count_doc_terms, context_counts_per_doc,
)
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    locate_terms_in_docs,
)
from graded_readers_stats.tfidf import calc_mean_doc_idfs, \
    calc_mean_doc_context_idfs
from graded_readers_stats.tree import (
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
        texts_df["IDF"] = calc_mean_doc_idfs(docs_locs)
        texts_df['TFIDF'] = texts_df["Freq"] * texts_df["IDF"]
        texts_df = texts_df.drop(columns="IDF")

    with Timer(name='Tree', text=timer_text):
        texts_df["Tree"] = texts_tree_props_pipeline(storage, docs_locs)

##############################################################################
#                                Contexts                                    #
##############################################################################

    with Timer(name='Context collect', text=timer_text):
        context_words_by_terms = collect_context_words_by_terms(
            terms_locs,
            texts,
            window=3
        )
        # For debugging only
        # context_words_by_docs = collect_context_words_by_docs(
        #     docs_locs,
        #     texts,
        #     window=3
        # )
        # texts_df['Context words'] = context_words_by_docs

    with Timer(name='Context locate terms', text=timer_text):
        ctxs_locs_by_term_indices = locate_ctx_terms_in_docs(
            context_words_by_terms,
            texts
        )
        ctxs_locs_by_terms = to_list_of_dicts(
            context_words_by_terms,
            ctxs_locs_by_term_indices
        )
        terms_count = len(terms_df)
        docs_count = len(texts_df)
        ctxs_locs_by_docs = transpose_ctx_terms_to_docs_locations(
            ctxs_locs_by_terms,
            terms_count,
            docs_count,
        )

    with Timer(name='Context frequency', text=timer_text):
        context_counts_by_doc = context_counts_per_doc(ctxs_locs_by_docs)
        # For debugging only
        # texts_df['Context count per word'] = context_counts_by_doc
        texts_df['Context count'] = list(map(avg, context_counts_by_doc))
        texts_df['Context freq'] = \
            texts_df['Context count'] / texts_df['Total']

    with Timer(name='Context TFIDF', text=timer_text):
        texts_df["Context IDF"] = calc_mean_doc_context_idfs(ctxs_locs_by_docs)
        texts_df['Context TFIDF'] = texts_df["Context freq"] * texts_df["Context IDF"]
        texts_df = texts_df.drop(columns="Context IDF")

    with Timer(name='Context Tree', text=timer_text):
        print("breakpoint")
        # terms_df['Context tree'] = list(trees_pipeline(storage)(ctxs_locs))
        # texts_df["Context tree"] = calc_mean_doc_context_tree(
        #     storage,
        #     docs_locs
        # )

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
#     print('ANALYZE DOCS END'
