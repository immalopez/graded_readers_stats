###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

import time

from graded_readers_stats import data, frequency, preprocess, tree
from graded_readers_stats.constants import (
    COL_READERS_LOCATIONS
)

start = time.time()

# Load data
vocabulary, readers, literature = data.load(trial=True)

# Preprocess data
vocabulary = preprocess.run(vocabulary, preprocess.vocabulary_pipeline)
readers = preprocess.run(readers, preprocess.text_analysis_pipeline)
literature = preprocess.run(literature, preprocess.text_analysis_pipeline)

preprocess.find_phrases_in_texts(vocabulary, readers, COL_READERS_LOCATIONS)
preprocess.count_stuff(vocabulary, readers)

# preprocess.collect_context_for_phrases_in_texts(
#     vocabulary, readers, column_prefix=PREFIX_READER
# )
# readers_by_level = readers.groupby(COL_LEVEL)

# # Vocabulary Frequencies
# frequency.count_vocab_in_sentences_by_groups(
#     vocabulary, readers_by_level, column_prefix=PREFIX_READER
# )
#
# # Vocabulary's Context Frequencies
# frequency.count_vocab_context_in_sentences_by_groups(
#     vocabulary, readers_by_level, column_prefix=PREFIX_READER_CONTEXT
# )
#
# # Tree widths and depths
# tree.get_tree_widths_and_depths(
#     vocabulary,
#     readers_by_level
# )

data.save(vocabulary, readers, literature)

print('DONE!')
print('Total time: %d secs' % (time.time() - start))


###############################################################################
#                                 PROFILING                                   #
###############################################################################

# wrap the code above in a 'main()' function and call it
# if __name__ == '__main__':
#     main()

# import profile
# profile.run('main()', 'profile_stats')
#
# import pstats
# from pstats import SortKey
# p = pstats.Stats('profile_stats')
# p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
