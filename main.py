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

preprocess.vocabs_locations_in_texts(vocab, reader, READER)
preprocess.vocabs_locations_in_texts(vocab, litera, LITERA)
# preprocess.print_words_at_locations(vocabs, reader)

reader_by_level = reader.groupby(COL_LEVEL)
litera_by_level = litera.groupby(COL_LEVEL)

# Vocabulary Frequencies
frequency.count_vocab_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER
)
frequency.count_vocab_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA
)
frequency.total_count_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER
)
frequency.total_count_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA
)
frequency.frequency_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER
)
frequency.frequency_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA
)

# Vocabulary's Context Frequencies
preprocess.collect_all_vocab_contexts_in_texts(
    vocab, reader, column=READER
)
preprocess.collect_all_vocab_contexts_in_texts(
    vocab, litera, column=LITERA
)
preprocess.vocabs_locations_in_texts(vocab, reader, READER, is_context=True)
preprocess.vocabs_locations_in_texts(vocab, litera, LITERA, is_context=True)
# preprocess.print_words_at_locations(vocab, reader, is_context=True)
frequency.count_vocab_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER, is_context=True
)
frequency.count_vocab_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA, is_context=True
)
frequency.total_count_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER, is_context=True
)
frequency.total_count_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA, is_context=True
)
frequency.frequency_in_texts_grouped_by_level(
    vocab, reader_by_level, column=READER, is_context=True
)
frequency.frequency_in_texts_grouped_by_level(
    vocab, litera_by_level, column=LITERA, is_context=True
)

# frequency.count_vocab_context_in_sentences_by_groups(
#     vocab, reader_by_level, column=READER
# )

# Tree widths and depths
tree.make_trees_for_occurrences(vocab, reader, column=READER)
tree.make_trees_for_occurrences(vocab, litera, column=LITERA)

tree.calculate_tree_props(vocab, column=READER)
tree.calculate_tree_props(vocab, column=LITERA)

# vocabs.drop(columns=['Readers_locations'], inplace=True)
# data.save(vocabs, reader, litera)

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
