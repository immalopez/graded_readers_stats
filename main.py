import argparse

import cmd_analyze

parser = argparse.ArgumentParser()
# parser.add_argument('terms_path',
#                     help='file path to a CSV with terms/vocabulary')
# parser.add_argument('texts_path',
#                     help='file path to a CSV with file paths to texts')
# parser.add_argument('--level',
#                     help='only process texts of the specified level column'
#                          'e.g. --level=Inicial would process only rows with a'
#                          'value Inicial for the column Level that is assumed'
#                          'to be present in all loaded texts.')
#
# args = parser.parse_args()
# print(args)

print()
print('MAIN START')
cmd_analyze.analyze()
print('MAIN END')
