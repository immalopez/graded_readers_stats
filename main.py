from pandas.core.common import flatten
from graded_readers_stats import data, frequency, preprocess, tree, stats
from graded_readers_stats.constants import (
    COL_LEVEL,
    LOCATIONS,
    FREQUENCY,
    COL_LEMMA,
    TFIDF,
)

# Steps
# -
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
# TODO: Refactor to not return column
# TODO: Optimize search to not always look for sub-lists
vocab[LOCATIONS] = preprocess.vocabs_locations_in_texts(vocab, group_1, '')[1]

# Frequency
found_words = vocab[LOCATIONS].apply(lambda x: sum(1 for d in x for _ in d))
total_words = sum(1 for _ in flatten(group_1[COL_LEMMA]))
vocab[FREQUENCY] = found_words / total_words

# TFIDF
vocab[TFIDF] = frequency.tfidfs(vocab_locs=vocab[LOCATIONS], docs=group_1)

# Tree

# Search for context
# Frequency
# TFIDF
# Tree

print("Breakpoint")
