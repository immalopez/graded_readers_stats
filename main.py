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
    locate_terms_in_docs,
)

timer_text = '{name}: {:0.0f} seconds'
start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

with Timer(name='Load data', text=timer_text):
    trial, use_cache = True, False
    terms_df = data.load(data.Dataset.VOCABULARY, trial, use_cache)
    readers = data.load(data.Dataset.READERS, trial, use_cache)

with Timer(name='Group', text=timer_text):
    reader_by_level = readers.groupby(COL_LEVEL)
    texts_df = reader_by_level.get_group('Inicial').reset_index(drop=True)

# TODO: Delete redundant columns from DataFrame
with Timer(name='Preprocess', text=timer_text):
    terms_df = run(terms_df, vocabulary_pipeline)
    texts_df = run(texts_df, text_analysis_pipeline)
    texts = texts_df[COL_LEMMA]
    st_docs = texts_df[COL_STANZA_DOC]

##############################################################################
#                               Process Terms                                #
##############################################################################

with Timer(name='Locate terms', text=timer_text):
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
    terms_locs = locate_terms_in_docs(terms, texts)

with Timer(name='Frequency', text=timer_text):
    found_words = [sum(1 for doc in docs for _ in doc) for docs in terms_locs]
    total_words = sum(1 for _ in flatten(texts))
    terms_df[FREQUENCY] = [count / total_words for count in found_words]

with Timer(name='TFIDF', text=timer_text):
    terms_df[TFIDF] = tfidfs(term_locs=terms_locs, docs=texts)

with Timer(name='Trees', text=timer_text):
    trees = tree.make_trees_for_occurrences_v2(
        term_locs=terms_locs, stanza_docs=st_docs
    )
    terms_df['Trees'] = tree.calculate_tree_props_v2(trees)


##############################################################################
#                             Process Contexts                               #
##############################################################################

with Timer(name='Context collect', text=timer_text):
    ctx_by_term = context.collect_context_words_multiple(
        terms_locs, texts, window=3
    )

with Timer(name='Context locate terms', text=timer_text):
    ctx_terms_flat = list(flatten(ctx_by_term))
    ctx_terms_wrap = [[word] for word in ctx_terms_flat]
    ctx_terms_locs = locate_terms_in_docs(ctx_terms_wrap, texts)
    ctx_term_loc_dict = dict(zip(ctx_terms_flat, ctx_terms_locs))

with Timer(name='Context frequency', text=timer_text):
    ctx_terms_2 = ctx_by_term[2]
    ctx_terms_2_locs = map(ctx_term_loc_dict.__getitem__, ctx_terms_2)
    found_words = [sum(1 for doc in docs for _ in doc) for docs in ctx_terms_2_locs]
    total_words = sum(1 for _ in flatten(texts))
    ctx_freqs = [count / total_words for count in found_words]

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
