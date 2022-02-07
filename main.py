import time

from codetiming import Timer
from funcy import *
from pandas.core.common import flatten

import graded_readers_stats.context
from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
    FREQUENCY,
    TFIDF,
)
from graded_readers_stats.context import collect_context_words
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
#                               Process Terms                                #
##############################################################################

with Timer(name='Locate terms', text=timer_text):
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
    terms_locs = locate_terms_in_docs(terms, texts)

with Timer(name='Frequency', text=timer_text):
    terms_df[FREQUENCY] = freqs_by_term(terms_locs, words_total)

with Timer(name='TFIDF', text=timer_text):
    terms_df[TFIDF] = tfidfs(terms_locs, texts)

with Timer(name='Trees', text=timer_text):
    terms_df['Trees'] = tree_props_pipeline(terms_locs, st_docs)


##############################################################################
#                             Process Contexts                               #
##############################################################################

with Timer(name='Context collect', text=timer_text):
    ctx_by_term = collect_context_words(terms_locs, texts, window=3)

with Timer(name='Context locate terms', text=timer_text):
    ctx_terms_flat = list(flatten(ctx_by_term))
    ctx_terms_wrap = [[word] for word in ctx_terms_flat]
    ctx_terms_locs = locate_terms_in_docs(ctx_terms_wrap, texts)
    ctx_term_loc_dict = dict(zip(ctx_terms_flat, ctx_terms_locs))

with Timer(name='Context frequency', text=timer_text):
    # Pseudocode
    # for every term (map?)
    #   ctx_words = get context words for term
    #   ctx_locs = map words to locations
    #   ctx_counts = count(ctx_locs)
    #   ctx_freqs = counts / total_words
    #   avg_freq = sum(ctx_freqs) / len(ctx_freqs)

    # Alternative using pipelines
    # pipeline_freq = rcompose(
    #   ctx_words_for_term,
    #   partial(map, ctx_word_to_loc),
    #   ctx_locs_to_counts,
    # )
    # ctx_freqs_by_term = pipeline_freq(terms)

    ctx_terms_2 = ctx_by_term[2]
    ctx_terms_2_locs = map(ctx_term_loc_dict.__getitem__, ctx_terms_2)
    words_found = [sum(1 for doc in docs for _ in doc) for docs in ctx_terms_2_locs]
    words_total = sum(1 for _ in flatten(texts))
    ctx_freqs = [count / words_total for count in words_found]

with Timer(name='Context TFIDF', text=timer_text):
    # select relevant words
    pass

with Timer(name='Context Tree', text=timer_text):
    # select relevant words
    pass

with Timer(name='MSTTR for all text', text=timer_text):
    pass

utils.duration(start_main, 'Main')
print('')
