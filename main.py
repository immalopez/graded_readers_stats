import argparse

import cmd_analyze
import cmd_merge_output

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

##############################################################################
#                                 Analyze                                    #
##############################################################################

subparser_analyze = subparsers.add_parser(
    'analyze',
    help='Calculate Freq, TFIDF and Tree')
subparser_analyze.add_argument(
    'vocabulary_path',
    help='file path to a CSV with terms/vocabulary')
subparser_analyze.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser_analyze.add_argument(
    'level',
    help='only process texts of the specified level '
         'column e.g. --level=Inicial would process '
         'only rows with a value Inicial for the column '
         'Level that is assumed to be present in all '
         'loaded texts.')
subparser_analyze.set_defaults(func=cmd_analyze.analyze)

##############################################################################
#                          Download Native Corpus
##############################################################################

# TODO: Add subparser for handling native corpus

##############################################################################
#                               Merge output
##############################################################################

subparser_analyze = subparsers.add_parser(
    'merge-output',
    help='Merge all CSV files into a single main.csv')
subparser_analyze.set_defaults(func=cmd_merge_output.merge_output)


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
