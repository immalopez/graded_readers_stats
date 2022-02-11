import time

from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import config
from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
)
from graded_readers_stats.context import (
    collect_context_words,
    freqs_pipeline,
    tfidfs_pipeline, trees_pipeline, locate_ctx_terms_in_docs,
)
from graded_readers_stats.data import load, Dataset
from graded_readers_stats.frequency import freqs_by_term, tfidfs
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    locate_terms_in_docs,
)
from graded_readers_stats.stats import get_msttr
from graded_readers_stats.tree import tree_props_pipeline

config.is_debug = False

timer_text = '{name}: {:0.0f} seconds'
start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

with Timer(name='Load data', text=timer_text):
    trial, use_cache = False, False
    terms_df = load(Dataset.VOCABULARY, trial, use_cache)
    # terms_df = terms_df[:5]
    readers = load(Dataset.READERS, trial, use_cache)

with Timer(name='Group', text=timer_text):
    reader_by_level = readers.groupby(COL_LEVEL)
    texts_df = reader_by_level.get_group('Inicial').reset_index(drop=True)
    # texts_df = texts_df[:int(len(texts_df)/2)]

# TODO: Delete redundant columns from DataFrame
with Timer(name='Preprocess', text=timer_text):
    terms_df = run(terms_df, vocabulary_pipeline)
    texts_df = run(texts_df, text_analysis_pipeline)
    texts = texts_df[COL_LEMMA]
    storage = {
        'stanza': texts_df[COL_STANZA_DOC],
        'tree': {}
    }
    num_words = sum(1 for _ in flatten(texts))
    terms_df.drop(columns=COL_STANZA_DOC, inplace=True)

##############################################################################
#                                 Terms                                      #
##############################################################################

with Timer(name='Locate terms', text=timer_text):
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
    terms_locs = locate_terms_in_docs(terms, texts)

with Timer(name='Frequency', text=timer_text):
    terms_df['Frequency'] = freqs_by_term(terms_locs, num_words)

with Timer(name='TFIDF', text=timer_text):
    terms_df['TFIDF'] = tfidfs(terms_locs, texts)

with Timer(name='Tree', text=timer_text):
    terms_df['Tree'] = tree_props_pipeline(storage, terms_locs)


##############################################################################
#                                Contexts                                    #
##############################################################################

with Timer(name='Context collect', text=timer_text):
    ctx_words_by_term = collect_context_words(terms_locs, texts, window=3)
    terms_df['Context words'] = ctx_words_by_term

with Timer(name='Context locate terms', text=timer_text):
    ctxs_locs = locate_ctx_terms_in_docs(ctx_words_by_term, texts)

with Timer(name='Context frequency', text=timer_text):
    terms_df['Context frequency'] = list(freqs_pipeline(num_words)(ctxs_locs))

with Timer(name='Context TFIDF', text=timer_text):
    terms_df['Context TFIDF'] = list(tfidfs_pipeline(texts)(ctxs_locs))

with Timer(name='Context Tree', text=timer_text):
    terms_df['Context Tree'] = list(trees_pipeline(storage)(ctxs_locs))

##############################################################################
#                                  Others                                    #
##############################################################################

with Timer(name='MSTTR', text=timer_text):
    joined_text = ' '.join(texts_df['Raw text'])
    print(f'{get_msttr(joined_text)}')

utils.duration(start_main, 'main')
print('')
