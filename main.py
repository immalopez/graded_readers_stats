import time

from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import (
    context,
    data,
    tree,
    utils,
)
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
    FREQUENCY,
    TFIDF,
)
from graded_readers_stats.frequency import (
    tfidfs
)
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    find_term_locations_in_docs,
)

timer_text = '{name}: {:0.0f} seconds'

start_main = time.time()
trial = True
use_cache = False

# Load data
with Timer(name='Load data', text=timer_text):
    terms_df = data.load(data.Dataset.VOCABULARY, trial, use_cache)
    readers = data.load(data.Dataset.READERS, trial, use_cache)

# Group
with Timer(name='Group', text=timer_text):
    reader_by_level = readers.groupby(COL_LEVEL)
    texts_df = reader_by_level.get_group('Inicial').reset_index(drop=True)

# Preprocess
# TODO: Move non-output columns to local var
with Timer(name='Preprocess', text=timer_text):
    terms_df = run(terms_df, vocabulary_pipeline)
    texts_df = run(texts_df, text_analysis_pipeline)

##############################################################################
#                               Process Terms                                #
##############################################################################

with Timer(name='Locate terms', text=timer_text):
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
    terms_locs = find_term_locations_in_docs(terms, texts_df[COL_LEMMA])

with Timer(name='Frequency', text=timer_text):
    found_words = [sum(1 for doc in docs for _ in doc) for docs in terms_locs]
    total_words = sum(1 for _ in flatten(texts_df[COL_LEMMA]))
    terms_df[FREQUENCY] = [count / total_words for count in found_words]

with Timer(name='TFIDF', text=timer_text):
    terms_df[TFIDF] = tfidfs(term_locs=terms_locs, docs=texts_df[COL_LEMMA])

with Timer(name='Trees', text=timer_text):
    trees = tree.make_trees_for_occurrences_v2(
        term_locs=terms_locs, stanza_docs=texts_df[COL_STANZA_DOC]
    )
    terms_df['Trees'] = tree.calculate_tree_props_v2(trees)


##############################################################################
#                                Contexts                                    #
##############################################################################

# Collect context
contexts_by_term = context.collect_context_words_multiple(
    terms_locs, texts_df[COL_LEMMA], window=3
)

# Locate context
# - unified context list
ctx_terms_flat = list(flatten(contexts_by_term))
ctx_terms = [[word] for word in ctx_terms_flat]
ctx_terms_locs = find_term_locations_in_docs(ctx_terms, texts_df[COL_LEMMA])
ctx_term_loc_map = dict(zip(ctx_terms_flat, ctx_terms_locs))

# Frequency context
# - select relevant words
start = time.time()

ctx_terms_2 = contexts_by_term[2]
ctx_terms_2_locs = map(ctx_term_loc_map.__getitem__, ctx_terms_2)


found_words = [sum(1 for doc in docs for _ in doc) for docs in ctx_terms_2_locs]
total_words = sum(1 for _ in flatten(texts_df[COL_LEMMA]))
ctx_freqs = [count / total_words for count in found_words]
utils.duration(start, 'Context Frequency')

# TFIDF context
# - select relevant words

# Tree context
# - select relevant words

# MSTTR for all text

utils.duration(start_main, 'Main')
print('')
