from pandas.core.common import flatten
from graded_readers_stats import data, frequency, preprocess, tree, stats
from graded_readers_stats.constants import (
    COL_LEVEL,
    COL_STANZA_DOC,
    FREQUENCY,
    COL_LEMMA,
    TFIDF,
)

# Load data
trial = True
use_cache = False
vocab = data.load(data.Dataset.VOCABULARY, trial, use_cache)
readers = data.load(data.Dataset.READERS, trial, use_cache)

# Group
reader_by_level = readers.groupby(COL_LEVEL)
group_1 = reader_by_level.get_group('Inicial').reset_index(drop=True)

# Preprocess
vocab = preprocess.run(vocab, preprocess.vocabulary_pipeline)
group_1 = preprocess.run(group_1, preprocess.text_analysis_pipeline)

# Search for vocab
# TODO: Optimize search to not always look for sub-lists
terms = [term for terms in vocab[COL_LEMMA] for term in terms]
locations = preprocess.find_term_locations_in_docs(terms, group_1[COL_LEMMA])

# Frequency
found_words = [sum(1 for doc in docs for _ in doc) for docs in locations]
total_words = sum(1 for _ in flatten(group_1[COL_LEMMA]))
vocab[FREQUENCY] = [count / total_words for count in found_words]

# TFIDF
vocab[TFIDF] = frequency.tfidfs(vocab_locs=locations, docs=group_1)

# Tree
trees = tree.make_trees_for_occurrences_v2(
    term_locs=locations, stanza_docs=group_1[COL_STANZA_DOC]
)
vocab['Trees'] = tree.calculate_tree_props_v2(trees)

# Search for context
# Frequency
# TFIDF
# Tree

print("Breakpoint")
