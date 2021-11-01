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
    vocabulary, readers_by_level, column_prefix='Reader_'
)

# Vocabulary's Context Frequencies
frequency.collect_context_for_phrases_in_texts(
    vocabulary, readers, column_prefix='Reader_'
)
frequency.count_context_in_sentences_by_groups(
    vocabulary, readers_by_level, column_prefix='Reader_Context_'
)
