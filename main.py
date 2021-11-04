###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################

import time

from graded_readers_stats import data, frequency, preprocess, tree
from graded_readers_stats.constants import *

start = time.time()

# Load data
vocab, reader, litera = data.load(trial=True, use_cache=True)

# Preprocess data
vocab = preprocess.run(vocab, preprocess.vocabulary_pipeline)
reader = preprocess.run(reader, preprocess.text_analysis_pipeline)
litera = preprocess.run(litera, preprocess.text_analysis_pipeline)

preprocess.find_phrases_in_texts(vocab, reader, READER + ' ' + LOCATIONS)
preprocess.find_phrases_in_texts(vocab, litera, LITERA + ' ' + LOCATIONS)
# preprocess.print_words_at_locations(vocab, reader)

# preprocess.collect_context_for_phrases_in_texts(
#     vocab, reader, column_prefix=DATASET_READER
# )

reader_by_level = reader.groupby(COL_LEVEL)
litera_by_level = litera.groupby(COL_LEVEL)

# Vocabulary Frequencies
frequency.count_vocab_in_sentences_by_groups_v1(
    vocab, reader_by_level, column=READER
)
frequency.count_vocab_in_sentences_by_groups_v1(
    vocab, litera_by_level, column=LITERA
)

# # Vocabulary's Context Frequencies
# frequency.count_vocab_context_in_sentences_by_groups(
#     vocab, reader_by_level, column_prefix=PREFIX_READER_CONTEXT
# )
#
# # Tree widths and depths
# tree.get_tree_widths_and_depths(
#     vocab,
#     reader_by_level
# )

# vocab.drop(columns=['Readers_locations'], inplace=True)
data.save(vocab, reader, litera)

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
