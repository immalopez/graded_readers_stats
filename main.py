###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

# NOTE:
# Possible alternatives for the Spanish native corpus:
# https://crscardellino.ar/SBWCE/ AND
# https://www.cs.upc.edu/~nlp/wikicorpus/

from graded_readers_stats import data, preprocess, frequency

# Load data
vocabulary, readers, literature = data.load(trial=True)

# Preprocess data
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
literature = preprocess.run(literature, preprocess.text_analysis_pipeline)
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)

# Vocabulary Frequencies
readers_by_level = data.group_by_level(readers)
literat_by_level = data.group_by_level(literature)

frequency.count_occurrences(vocabulary, readers_by_level, 'Reader')
frequency.count_occurrences(vocabulary, literat_by_level, 'Literature')

# Vocabulary's Context Frequencies
frequency.get_context(vocabulary, readers_by_level, 'Reader')
frequency.get_context(vocabulary, literat_by_level, 'Literature')

frequency.count_context_occurrences(vocabulary, readers_by_level, 'Reader')
frequency.count_context_occurrences(vocabulary, literat_by_level, 'Literature')
