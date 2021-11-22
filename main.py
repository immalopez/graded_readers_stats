###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################
import time
from multiprocessing import Pool
from os import cpu_count

from graded_readers_stats import data, frequency, preprocess, tree
from graded_readers_stats.constants import (
    READER,
    LITERA,
    NATIVE,
    COL_LEVEL,
)


def calculatestar(args):
    return calculate(*args)


def calculate(func, args):
    return func(*args)


def main():
    start = time.time()

    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        # Load data
        data_tasks = [(data.load, (ds, True, False)) for ds in data.Dataset]
        vocab, reader, litera, native  = pool.map(calculatestar, data_tasks)

        # Preprocess data
        process_tasks = [
            (preprocess.run, (vocab, preprocess.vocabulary_pipeline)),
            (preprocess.run, (reader, preprocess.text_analysis_pipeline)),
            (preprocess.run, (litera, preprocess.text_analysis_pipeline)),
            (preprocess.run, (native, preprocess.text_analysis_pipeline)),
        ]
        vocab, reader, litera, native = pool.map(calculatestar, process_tasks)

        # Find vocab in texts
        location_tasks = [
            (preprocess.vocabs_locations_in_texts, (vocab, reader, READER)),
            (preprocess.vocabs_locations_in_texts, (vocab, litera, LITERA)),
            (preprocess.vocabs_locations_in_texts, (vocab, native, NATIVE)),
        ]
        location_columns = pool.map(calculatestar, location_tasks)
        for name, locations in location_columns:
            vocab[name] = locations

        reader_by_level = reader.groupby(COL_LEVEL)
        litera_by_level = litera.groupby(COL_LEVEL)
        native_by_level = native.groupby(COL_LEVEL)

        # Vocabulary Frequencies
        count_tasks = [
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER)),
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA)),
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE)),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER)),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA)),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE)),
        ]
        counts = pool.map(calculatestar, count_tasks)
        for name, values in counts:
            vocab[name] = values

        # Frequencies
        freqs_tasks = [
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE)),
        ]
        freqs = pool.map(calculatestar, freqs_tasks)
        for name, freq in freqs:
            vocab[name] = freq

        # Vocabulary's Context Frequencies
        collect_task = [
            (preprocess.collect_all_vocab_contexts_in_texts,
             (vocab, reader, READER)),
            (preprocess.collect_all_vocab_contexts_in_texts,
             (vocab, litera, LITERA)),
            (preprocess.collect_all_vocab_contexts_in_texts,
             (vocab, native, NATIVE)),
        ]
        context_texts = pool.map(calculatestar, collect_task)
        for name, contexts in context_texts:
            vocab[name] = contexts

        location_tasks = [
            (preprocess.vocabs_locations_in_texts,
             (vocab, reader, READER, True)),
            (preprocess.vocabs_locations_in_texts,
             (vocab, litera, LITERA, True)),
            (preprocess.vocabs_locations_in_texts,
             (vocab, native, NATIVE, True)),
        ]
        locations = pool.map(calculatestar, location_tasks)
        for name, locs in locations:
            vocab[name] = locs

        count_tasks = [
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER, True)),
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA, True)),
            (frequency.count_vocab_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE, True )),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER, True)),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA, True)),
            (frequency.total_count_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE, True)),
        ]
        counts = pool.map(calculatestar, count_tasks)
        for name, values in counts:
            vocab[name] = values

        # Context Frequencies
        freqs_tasks = [
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER, True)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA, True)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE, True)),
        ]
        freqs = pool.map(calculatestar, freqs_tasks)
        for name, freq in freqs:
            vocab[name] = freq

        # Tree widths and depths
        tree_tasks = [
            (tree.make_trees_for_occurrences, (vocab, reader, READER)),
            (tree.make_trees_for_occurrences, (vocab, litera, LITERA)),
            (tree.make_trees_for_occurrences, (vocab, native, NATIVE)),
        ]
        trees = pool.map(calculatestar, tree_tasks)
        for name, tr in trees:
            vocab[name] = tr

        tree_prop_tasks = [
            (tree.calculate_tree_props, (vocab, READER)),
            (tree.calculate_tree_props, (vocab, LITERA)),
            (tree.calculate_tree_props, (vocab, NATIVE)),
        ]
        props = pool.map(calculatestar, tree_prop_tasks)
        for name, prop in props:
            vocab[name] = prop

        # vocabs.drop(columns=['Readers_locations'], inplace=True)
        # data.save(vocabs, reader, litera)

        print('DONE!')
        print('Total time: %d secs' % (time.time() - start))
        print()
        return vocab, reader, litera, native


###############################################################################
#                                 PROFILING                                   #
###############################################################################

if __name__ == '__main__':
    # vocab = main()
    vocab, reader, litera, native = main()
    print('Finished')
    # main()

# Uncomment to print profiling results
# if __name__ == '__main__':
#     import cProfile
#     import pstats
#     import io
#     from pstats import SortKey
#
#     with cProfile.Profile() as pr:
#         # Add to global variables for inspection in PyCharm
#         vocab, reader, litera, native = main()
#
#     output = io.StringIO()
#     ps = pstats.Stats(pr, stream=output).sort_stats(SortKey.CUMULATIVE)
#     ps.print_stats(10)
#     print(output.getvalue())
