import time

from codetiming import Timer

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.logit import logit
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline,
)


def execute(args):
    corpus_path = args.corpus_path
    max_docs = args.max_docs

    print()
    print('BAG-OF-WORDS START')
    print('---')
    print('corpus_path = ', corpus_path)
    print('max_docs = ', max_docs)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

    with Timer(name='Load data', text=timer_text):
        texts_df = read_pandas_csv(corpus_path)
        if max_docs:
            texts_df = texts_df[:max_docs]

    with Timer(name='Preprocess', text=timer_text):
        texts_df = run(texts_df, text_analysis_pipeline)
        texts = texts_df[COL_LEMMA]

##############################################################################
#                             Logistic Regression
##############################################################################

    with Timer(name='Logistic Regression', text=timer_text):
        logit(texts_df)

##############################################################################
#                                   Done
##############################################################################

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('BAG-OF-WORDS END')
