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

        print('DONE!')
        print('Total time: %d secs' % (time.time() - start))
        print()
        return vocab, reader, litera, native


    reader_by_level = reader.groupby(COL_LEVEL)
    litera_by_level = litera.groupby(COL_LEVEL)
    native_by_level = native.groupby(COL_LEVEL)

    # Vocabulary Frequencies
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER
    )
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA
    )
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE
    )

    # Vocabulary's Context Frequencies
    preprocess.collect_all_vocab_contexts_in_texts(
        vocab, reader, column=READER
    )
    preprocess.collect_all_vocab_contexts_in_texts(
        vocab, litera, column=LITERA
    )
    preprocess.collect_all_vocab_contexts_in_texts(
        vocab, native, column=NATIVE
    )
    preprocess.vocabs_locations_in_texts(vocab, reader, READER, is_context=True)
    preprocess.vocabs_locations_in_texts(vocab, litera, LITERA, is_context=True)
    preprocess.vocabs_locations_in_texts(vocab, native, NATIVE, is_context=True)
    # preprocess.print_words_at_locations(vocab, reader, is_context=True)
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER, is_context=True
    )
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA, is_context=True
    )
    frequency.count_vocab_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE, is_context=True
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER, is_context=True
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA, is_context=True
    )
    frequency.total_count_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE, is_context=True
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, reader_by_level, column=READER, is_context=True
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, litera_by_level, column=LITERA, is_context=True
    )
    frequency.frequency_in_texts_grouped_by_level(
        vocab, native_by_level, column=NATIVE, is_context=True
    )

    # frequency.count_vocab_context_in_sentences_by_groups(
    #     vocab, reader_by_level, column=READER
    # )

    # Tree widths and depths
    tree.make_trees_for_occurrences(vocab, reader, column=READER)
    tree.make_trees_for_occurrences(vocab, litera, column=LITERA)
    tree.make_trees_for_occurrences(vocab, native, column=NATIVE)

    tree.calculate_tree_props(vocab, column=READER)
    tree.calculate_tree_props(vocab, column=LITERA)
    tree.calculate_tree_props(vocab, column=NATIVE)

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
