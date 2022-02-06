import time

from pandas.core.common import flatten

from graded_readers_stats import (
    context,
    data,
    frequency,
    preprocess,
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

start_main = time.time()
trial = True
use_cache = False

# Load data
start = time.time()
vocab = data.load(data.Dataset.VOCABULARY, trial, use_cache)
readers = data.load(data.Dataset.READERS, trial, use_cache)
utils.duration(start, 'Loading')

# Group
reader_by_level = readers.groupby(COL_LEVEL)
group_1 = reader_by_level.get_group('Inicial').reset_index(drop=True)

# Preprocess
# TODO: Move non-output columns to local var
start = time.time()
vocab = preprocess.run(vocab, preprocess.vocabulary_pipeline)
group_1 = preprocess.run(group_1, preprocess.text_analysis_pipeline)
utils.duration(start, 'Preprocess')

# Locate terms
start = time.time()
terms = [term for terms in vocab[COL_LEMMA] for term in terms]
locations = preprocess.find_term_locations_in_docs(terms, group_1[COL_LEMMA])
utils.duration(start, 'Find terms')

# Frequency
start = time.time()
found_words = [sum(1 for doc in docs for _ in doc) for docs in locations]
total_words = sum(1 for _ in flatten(group_1[COL_LEMMA]))
vocab[FREQUENCY] = [count / total_words for count in found_words]
utils.duration(start, 'Frequency')

# TFIDF
start = time.time()
vocab[TFIDF] = frequency.tfidfs(term_locs=locations, docs=group_1[COL_LEMMA])
utils.duration(start, 'TFIDF')

# Tree
start = time.time()
trees = tree.make_trees_for_occurrences_v2(
    term_locs=locations, stanza_docs=group_1[COL_STANZA_DOC]
)
vocab['Trees'] = tree.calculate_tree_props_v2(trees)
utils.duration(start, 'Trees')

# Collect context
context_words = context.collect_context_words_multiple(
    locations, group_1[COL_LEMMA], window=3
)

# Locate context
# Frequency context
# TFIDF context
# Tree context

# MSTTR for all text

utils.duration(start_main, 'Main')
print('')
