###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

from graded_readers_stats import data, frequency, preprocess, tree

# Load data
vocabulary, readers = data.load(trial=True)

# Preprocess data
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)

# Vocabulary Frequencies
readers_by_level = readers.groupby(preprocess.LEVEL)
frequency.count_phrases_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix='Reader_'
)

# Vocabulary's Context Frequencies
# TODO: Move context collection to preprocess.py
frequency.collect_context_for_phrases_in_texts(
    vocabulary, readers, column_prefix='Reader_'
)
frequency.count_context_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix='Reader_Context_'
)

# Min and Max width
frequency.make_trees(vocabulary, readers_by_level, column_prefix='Reader_tree_')
tree.find_min_max_width(vocabulary, readers_by_level, column_prefix='Reader_')
tree.find_min_max_depth(vocabulary, readers_by_level, column_prefix='Reader_')
