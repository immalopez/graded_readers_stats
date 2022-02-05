from pandas.core.common import flatten
from graded_readers_stats import (
    data,
    frequency,
    preprocess,
    tree,
    utils,
    stats,
)
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
    FREQUENCY,
    TFIDF,
)
import time
start_main = time.time()

# Load data
start = time.time()
trial = True
use_cache = False
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

# Search for vocab
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
vocab[TFIDF] = frequency.tfidfs(vocab_locs=locations, docs=group_1)
utils.duration(start, 'TFIDF')

# Tree
start = time.time()
trees = tree.make_trees_for_occurrences_v2(
    term_locs=locations, stanza_docs=group_1[COL_STANZA_DOC]
)
vocab['Trees'] = tree.calculate_tree_props_v2(trees)
utils.duration(start, 'Trees')

# Search for context
# Frequency
# TFIDF
# Tree
# MSTTR

utils.duration(start_main, 'Main')
print('')
