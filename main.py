###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

from graded_readers_stats import data, frequency, preprocess

# Load data
vocabulary, readers = data.load(trial=True)

# Preprocess data
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)

# Vocabulary Frequencies
readers_by_level = readers.groupby(preprocess.LEVEL)
frequency.count_phrases_in_sentences_by_groups(
    vocabulary, readers_by_level, new_column_prefix='Reader_'
)

# Vocabulary's Context Frequencies
# frequency.get_context(vocabulary, readers_by_level, 'Reader')
# frequency.count_context_occurrences(vocabulary, readers_by_level, 'Reader')
