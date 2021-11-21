###############################################################################
#                      SPANISH GRADED READERS STATISTICS                      #
###############################################################################
import multiprocessing
from multiprocessing import Pool
from os import cpu_count
import time

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

    # vocab = data.load(
    #     dataset=data.Dataset.VOCABULARY,
    #     trial=True,
    #     use_cache=False
    # )
    # return vocab
    num_processes = cpu_count()
    tasks = [
        (data.load, (ds, True, True))
        for ds in data.Dataset
    ]
    # items = []
    # for t in tasks:
    #     items.append(calculate(*t))
    # print('DONE!')
    # print('Total time: %d secs' % (time.time() - start))
    # print()
    # vocab, reader, litera, native = items
    # return vocab, reader, litera, native

    with Pool(processes=num_processes) as pool:
        # result = pool.apply_async(calculate, args=(plus, (1, 2)))
        # result = pool.apply_async(plus, (1, 2))
        # print(result.get())
        # tasks = [pool.apply_async(plus, args=(i, 2)) for i in range(5)]
        # for t in tasks:
        #     print(t.get())
        # return 1, 2, 3, 4
        # return
        # results = [
        #     pool.apply_async(data.load, args=(ds, True, True))
        #     for ds in data.Dataset
        # ]
        # items = []
        # for r in results:
        #     items.append(r.get())
        # vocab, reader, litera, native = pool.map(
        #     printstuff,
        #     ['a', 'b', 'c', 'd']
        #     # [(dataset, True, True) for dataset in data.Dataset]
        # )
        vocab, reader, litera, native  = pool.map(calculatestar, tasks)
        # vocab, reader, litera, native = items
        print('DONE!')
        print('Total time: %d secs' % (time.time() - start))
        print()
        return vocab, reader, litera, native

    # Load data
    vocab, reader, litera, native = data.load(trial=False, use_cache=True)
    print('DONE!')
    print('Total time: %d secs' % (time.time() - start))
    print()
    return vocab, reader, litera, native

    # Preprocess data
    vocab = preprocess.run(vocab, preprocess.vocabulary_pipeline)
    reader = preprocess.run(reader, preprocess.text_analysis_pipeline)
    litera = preprocess.run(litera, preprocess.text_analysis_pipeline)
    native = preprocess.run(native, preprocess.text_analysis_pipeline)

    preprocess.vocabs_locations_in_texts(vocab, reader, READER)
    preprocess.vocabs_locations_in_texts(vocab, litera, LITERA)
    preprocess.vocabs_locations_in_texts(vocab, native, NATIVE)
    # preprocess.print_words_at_locations(vocabs, reader)

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
