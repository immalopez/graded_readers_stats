import argparse

from graded_readers_stats.commands import (
    cmd_download_native_corpus,
    cmd_merge_output,
    cmd_analyze,
    cmd_bag_of_words
)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

##############################################################################
#                          Download Native Corpus
##############################################################################

subparser = subparsers.add_parser(
    'download-native-corpus',
    help='Download cess_esp file from NLTK and generate text and csv files')
subparser.set_defaults(
    func=cmd_download_native_corpus.download_native_corpus
)

##############################################################################
#                                 Analyze                                    #
##############################################################################

subparser = subparsers.add_parser(
    'analyze',
    help='Calculate Freq, TFIDF and Tree')
subparser.add_argument(
    'vocabulary_path',
    help='file path to a CSV with terms/vocabulary')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    'level',
    help='only process texts of the specified level '
         'column e.g. --level=Inicial would process '
         'only rows with a value Inicial for the column '
         'Level that is assumed to be present in all '
         'loaded texts.')
subparser.add_argument('--max_terms', type=int,
                       help='max number of terms to analyze. Useful for tests')
subparser.add_argument('--max_docs', type=int,
                       help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_analyze.analyze)

##############################################################################
#                               Bag of Words                                 #
##############################################################################

subparser = subparsers.add_parser(
    'bag-of-words',
    help='Predict level using BoW')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    'level',
    help='only process texts of the specified level '
         'column e.g. --level=Inicial would process '
         'only rows with a value Inicial for the column '
         'Level that is assumed to be present in all '
         'loaded texts.')
subparser.add_argument('--max_docs', type=int,
                       help='max number of docs to analyze. Useful for tests')
subparser.add_argument('--shorten-content', action="store_true",
                       help='shortens content of docs for faster processing.')
subparser.add_argument('--strip-named-entities', action='store_true',
                       help='detect and strip Named Entities using Stanza.')
subparser.set_defaults(func=cmd_bag_of_words.execute)

##############################################################################
#                               Merge output
##############################################################################

subparser = subparsers.add_parser(
    'merge-output',
    help='Merge all CSV files into a single main.csv')
subparser.set_defaults(func=cmd_merge_output.merge_output)

##############################################################################
#                                  MAIN
##############################################################################

print()
print('---')
print()
print('Parsing arguments...')
args = parser.parse_args()

print('Parsing done:', args)
args.func(args)
