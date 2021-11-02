###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

from graded_readers_stats import data, frequency, preprocess, tree
from graded_readers_stats.constants import *


# Load data
vocabulary, readers = data.load(trial=True)

# Preprocess data
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)
preprocess.collect_context_for_phrases_in_texts(
    vocabulary, readers, column_prefix=PREFIX_READER
)
readers_by_level = readers.groupby(COL_LEVEL)

# Vocabulary Frequencies
frequency.count_vocab_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix=PREFIX_READER
)

# Vocabulary's Context Frequencies
frequency.count_vocab_context_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix=PREFIX_READER_CONTEXT
)

# Tree widths and depths
tree.get_tree_widths_and_depths(
    vocabulary,
    readers_by_level
)
