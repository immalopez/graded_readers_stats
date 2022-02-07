import time

from codetiming import Timer
from funcy import *
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.stats import get_msttr
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
)
from graded_readers_stats.context import (
    avg,
    avg_tuples,
    collect_context_words,
    ctxs_locs_by_term,
)
from graded_readers_stats.data import load, Dataset
from graded_readers_stats.frequency import freqs_by_term, tfidfs
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    locate_terms_in_docs,
)
from graded_readers_stats.tree import tree_props_pipeline

timer_text = '{name}: {:0.0f} seconds'
start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

with Timer(name='Load data', text=timer_text):
    trial, use_cache = True, False
    terms_df = load(Dataset.VOCABULARY, trial, use_cache)
    readers = load(Dataset.READERS, trial, use_cache)

with Timer(name='Group', text=timer_text):
    reader_by_level = readers.groupby(COL_LEVEL)
    texts_df = reader_by_level.get_group('Inicial').reset_index(drop=True)

# TODO: Delete redundant columns from DataFrame
with Timer(name='Preprocess', text=timer_text):
    terms_df = run(terms_df, vocabulary_pipeline)
    texts_df = run(texts_df, text_analysis_pipeline)
    texts = texts_df[COL_LEMMA]
    st_docs = texts_df[COL_STANZA_DOC]
    words_total = sum(1 for _ in flatten(texts))

##############################################################################
#                                 Terms                                      #
##############################################################################

with Timer(name='Locate terms', text=timer_text):
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
    terms_locs = locate_terms_in_docs(terms, texts)

with Timer(name='Frequency', text=timer_text):
    terms_df['Frequency'] = freqs_by_term(terms_locs, words_total)

with Timer(name='TFIDF', text=timer_text):
    terms_df['TFIDF'] = tfidfs(terms_locs, texts)

with Timer(name='Trees', text=timer_text):
    terms_df['Trees'] = tree_props_pipeline(terms_locs, st_docs)


##############################################################################
#                                  Contexts                                  #
##############################################################################

with Timer(name='Context collect', text=timer_text):
    ctx_by_term = collect_context_words(terms_locs, texts, window=3)

with Timer(name='Context locate terms', text=timer_text):
    ctx_terms_flat = list(flatten(ctx_by_term))
    ctx_terms_wrap = [[word] for word in ctx_terms_flat]
    ctx_terms_locs = locate_terms_in_docs(ctx_terms_wrap, texts)
    ctx_term_loc_dict = dict(zip(ctx_terms_flat, ctx_terms_locs))

with Timer(name='Context frequency', text=timer_text):
    ctxs_locs = ctxs_locs_by_term(ctx_term_loc_dict, ctx_by_term)
    ctxs_freqs = [freqs_by_term(ctx_loc, words_total) for ctx_loc in ctxs_locs]
    ctx_freqs_by_term = [avg(freqs) for freqs in ctxs_freqs]
    terms_df['Context frequency'] = ctx_freqs_by_term

    # Alternative using pipelines
    # ctx_freqs_pipeline = rcompose(
    #     ctxs_locs_by_term,
    #     partial(map, rpartial(freqs_by_term, words_total)),
    #     partial(map, avg),
    # )
    # terms_df['Context frequency 2'] = list(
    #     ctx_freqs_pipeline(ctx_term_loc_dict, ctx_by_term)
    # )

with Timer(name='Context TFIDF', text=timer_text):
    tfidfs_pipeline = rcompose(
        ctxs_locs_by_term,
        partial(map, rpartial(tfidfs, texts)),
        partial(map, avg)
    )
    terms_df['Context TFIDF'] = list(
        tfidfs_pipeline(ctx_term_loc_dict, ctx_by_term)
    )

with Timer(name='Context Tree', text=timer_text):
    empty_tuple = (None, None, None, None, None, None)
    ctx_tree_pipeline = rcompose(
        ctxs_locs_by_term,
        partial(map, rpartial(tree_props_pipeline, st_docs)),
        partial(map, rpartial(avg_tuples, empty_tuple))
    )
    terms_df['Context Trees'] = list(
        ctx_tree_pipeline(ctx_term_loc_dict, ctx_by_term)
    )

with Timer(name='MSTTR', text=timer_text):
    joined_text = ' '.join(texts_df['Raw text'])
    print(f'{get_msttr(joined_text)}')

utils.duration(start_main, 'Main')
print('')
