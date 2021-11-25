###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################
import time
from multiprocessing import Pool

from graded_readers_stats import data, frequency, preprocess, tree
from graded_readers_stats.utils import calculatestar, NoPool
from graded_readers_stats.constants import (
    READER,
    LITERA,
    NATIVE,
    COL_LEVEL,
)


def main():
    trial = True
    use_cache = False

    start_main = time.time()

    print(f"Creating pool...")
    print()
    with NoPool() as pool:

        print('Loading data...')
        start = time.time()
        data_tasks = [
            (data.load, (ds, trial, use_cache)) for ds in data.Dataset
        ]
        vocab, reader, litera, native = pool.map(calculatestar, data_tasks)
        duration = time.time() - start
        print(f'Data loaded in {duration:.2f} seconds')
        print()

        print('Preprocessing data...')
        start = time.time()
        process_tasks = [
            (preprocess.run, (vocab, preprocess.vocabulary_pipeline)),
            (preprocess.run, (reader, preprocess.text_analysis_pipeline)),
            (preprocess.run, (litera, preprocess.text_analysis_pipeline)),
            (preprocess.run, (native, preprocess.text_analysis_pipeline)),
        ]
        vocab, reader, litera, native = pool.map(calculatestar, process_tasks)
        duration = time.time() - start
        print(f'Data preprocessed in {duration:.2f} seconds')
        print()

        print('Searching for vocabulary in texts...')
        start = time.time()
        location_tasks = [
            (preprocess.vocabs_locations_in_texts, (vocab, reader, READER)),
            (preprocess.vocabs_locations_in_texts, (vocab, litera, LITERA)),
            (preprocess.vocabs_locations_in_texts, (vocab, native, NATIVE)),
        ]
        location_columns = pool.map(calculatestar, location_tasks)
        for name, locations in location_columns:
            vocab[name] = locations
        duration = time.time() - start
        print(f'Vocabulary found in texts in {duration:.2f} seconds')
        print()

        print('Groping texts by level...')
        start = time.time()
        reader_by_level = reader.groupby(COL_LEVEL)
        litera_by_level = litera.groupby(COL_LEVEL)
        native_by_level = native.groupby(COL_LEVEL)
        duration = time.time() - start
        print(f'Texts grouped by level in {duration:.2f} seconds')
        print()

        print('Counting frequency of vocabulary in texts...')
        start = time.time()
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
        for result in counts:
            for name, values in result:
                vocab[name] = values
        duration = time.time() - start
        print(f'Frequency counted in texts in {duration:.2f} seconds')
        print()

        print('Calculating frequencies...')
        start = time.time()
        freqs_tasks = [
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE)),
        ]
        freqs = pool.map(calculatestar, freqs_tasks)
        for result in freqs:
            for name, freq in result:
                vocab[name] = freq
        duration = time.time() - start
        print(f'Frequencies calculated in {duration:.2f} seconds')
        print()

        print('Collecting context for vocabulary found in texts...')
        start = time.time()
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
        duration = time.time() - start
        print(f'Contexts collected in {duration:.2f} seconds')
        print()

        print('Searching for context in texts...')
        start = time.time()
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
        duration = time.time() - start
        print(f'Context words found in texts in {duration:.2f} seconds')
        print()

        print('Counting frequency of context in texts...')
        start = time.time()
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
        for result in counts:
            for name, values in result:
                vocab[name] = values
        duration = time.time() - start
        print(f'Context counted in texts in {duration:.2f} seconds')
        print()

        print('Calculating context frequencies...')
        start = time.time()
        freqs_tasks = [
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, reader_by_level, READER, True)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, litera_by_level, LITERA, True)),
            (frequency.frequency_in_texts_grouped_by_level,
             (vocab, native_by_level, NATIVE, True)),
        ]
        freqs = pool.map(calculatestar, freqs_tasks)
        for result in freqs:
            for name, freq in result:
                vocab[name] = freq
        duration = time.time() - start
        print(f'Frequencies calculated in {duration:.2f} seconds')
        print()

        print('Make trees for vocabulary...')
        start = time.time()
        tree_tasks = [
            (tree.make_trees_for_occurrences, (vocab, reader, READER)),
            (tree.make_trees_for_occurrences, (vocab, litera, LITERA)),
            (tree.make_trees_for_occurrences, (vocab, native, NATIVE)),
            # (tree.make_trees_for_occurrences_v2,
            #  (vocab, reader, READER, reader_by_level)),
            # (tree.make_trees_for_occurrences_v2,
            #  (vocab, litera, LITERA, litera_by_level)),
            # (tree.make_trees_for_occurrences_v2,
            #  (vocab, native, NATIVE, native_by_level)),
        ]
        trees = pool.map(calculatestar, tree_tasks)
        for name, tr in trees:
            vocab[name] = tr
        # for groups in trees:
        #     for group in groups:
        #         name, values = group
        #         vocab[name] = values
        duration = time.time() - start
        print(f'Trees made in {duration:.2f} seconds')
        print()

        print('Calculating tree properties...')
        start = time.time()
        tree_prop_tasks = [
            (tree.calculate_tree_props, (vocab, READER)),
            (tree.calculate_tree_props, (vocab, LITERA)),
            (tree.calculate_tree_props, (vocab, NATIVE)),
        ]
        props = pool.map(calculatestar, tree_prop_tasks)
        for name, prop in props:
            vocab[name] = prop
        duration = time.time() - start
        print(f'Tree properties calculated in {duration:.2f} seconds')
        print()

        print('Saving data...')
        start = time.time()
        # vocab.drop(columns=['Readers_locations'], inplace=True)
        # data.save(trial, vocab, reader, litera, native, None)
        duration = time.time() - start
        print(f'Data saved in {duration:.2f} seconds')
        print()

        print('Saving XLSX...')
        import xlsxwriter
        import pandas as pd
        writer = pd.ExcelWriter(
            'graded_readers_stats.xlsx',
            engine='xlsxwriter'
        )
        vocab.to_excel(writer)
        # writer.save()
        print('XLSX saved.')

        print('DONE!')
        print('Total time: %d secs' % (time.time() - start_main))
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
