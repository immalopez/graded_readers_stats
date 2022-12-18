import time
from collections import defaultdict

import pandas as pd
from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline,
)

timer_text = '{name}: {:0.0f} seconds'


def execute(args):
    corpus_paths = args.corpus_paths
    max_docs = args.max_docs

    print()
    print('STATS START')
    print('---')
    print('corpus_paths = ', corpus_paths)
    print('---')

    start_main = time.time()

    groups = [group
              for corpus_path in corpus_paths
              for group in _split_corpus_into_groups(corpus_path, max_docs)]
    results = [_calc_stats_for_group(group_df, max_docs)
               for group_df in groups]

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('STATS END')


def _split_corpus_into_groups(corpus_path: str, max_docs: int) -> list[pd.DataFrame]:

    with Timer(name=f'Load data', text=timer_text):
        texts = read_pandas_csv(corpus_path)

    with Timer(name=f'Group', text=timer_text):
        texts_groupedby = texts.groupby(COL_LEVEL)
        for group_name in texts_groupedby.groups:
            print("\n---")
            print("Processing group:", group_name)
            group_df = texts_groupedby\
                .get_group(group_name)\
                .reset_index(drop=True)\
                .copy()
            yield group_df


def _calc_stats_for_group(group_df: pd.DataFrame, max_docs: int):
    texts_df = group_df

    if max_docs:
        texts_df = texts_df[:max_docs].copy()

    with Timer(name='Preprocess', text=timer_text):
        texts_df = run(texts_df, text_analysis_pipeline)
        num_words = sum(1 for _ in flatten(texts_df[COL_LEMMA]))

    with Timer(name='UPOS', text=timer_text):
        stanza_docs = texts_df[COL_STANZA_DOC]
        all_words = [word
                     for doc in stanza_docs
                     for sent in doc.sentences
                     for word in sent.words]
        print('---')
        print()
        print('upos')
        print()
        stats_upos = defaultdict(   # upos
            lambda: defaultdict(    # feats
                int                 # count
            )
        )
        counts_upos = defaultdict(int)
        for w in all_words:
            feats = (w.feats or 'no_features').split('|')
            counts_upos[w.upos] += 1
            for f in feats:
                stats_upos[w.upos][f] += 1
        for key, value in sorted(stats_upos.items()):
            count = counts_upos[key]
            print(key, f'({count}, {count / num_words * 100:.2f}%)')
            if isinstance(value, defaultdict):
                for k, v in sorted(value.items()):
                    print(f'    {k} ({v}, {v / count * 100:.2f}%)')
            else:
                print(f'    {key}: {value}')

    with Timer(name='Dependency Relations', text=timer_text):
        print('---')
        print()
        print('deprel:')
        print()
        stats_deprel = defaultdict(int)
        for w in all_words:
            stats_deprel[w.deprel] += 1
        for k, v in sorted(stats_deprel.items()):
            print(f'{k} = {v}')

    return stats_upos
