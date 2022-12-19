from __future__ import annotations

import math
from itertools import zip_longest
import matplotlib.pyplot as plt
import time

import pandas as pd
from codetiming import Timer

from graded_readers_stats import utils
from graded_readers_stats.constants import (
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

    groups = [(group_name, group_df)
              for corpus_path in corpus_paths
              for group_name, group_df in load_groups(corpus_path, max_docs)]
    stats = {name: calc_stats_for_group(name, df, max_docs)
             for name, df in groups}
    group_names = [name for name, _ in groups]

    df = pd.DataFrame(
        {k: [stats[nn]["deprel"].setdefault(k, 0) for nn in group_names]
         for name in group_names
         for k in stats[name]["deprel"].keys()
         },
        index=group_names
    )

    ll = sorted(list(stats["Inicial"]["deprel"].keys()))
    for i in range(0, len(ll), 3):
        df.plot(kind="bar", y=ll[i:3+i])
        plt.xticks(rotation=0, ha='right')
        plt.savefig(f"output/stats/deprel-image-{i}.png")
        plt.show()

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('STATS END')


def load_groups(corpus_path: str, max_docs: int) -> list[pd.DataFrame]:

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
            yield group_name, group_df


def calc_stats_for_group(
        group_name: str,
        group_df: pd.DataFrame,
        max_docs: int
):
    texts_df = group_df

    if max_docs:
        texts_df = texts_df[:max_docs].copy()

    with Timer(name='Preprocess', text=timer_text):
        texts_df = run(texts_df, text_analysis_pipeline)

    with Timer(name='UPOS', text=timer_text):
        stanza_docs = texts_df[COL_STANZA_DOC]
        all_words = [word
                     for doc in stanza_docs
                     for sent in doc.sentences
                     for word in sent.words]
        all_words_count = len(all_words)

        upos_dict = {"count": 0, "vals": {}}
        for w in all_words:
            # count general words
            upos_dict["count"] += 1

            # init and count the current upos
            curr_upos = upos_dict["vals"]\
                .setdefault(w.upos, {"count": 0})
            curr_upos["count"] += 1

            if w.feats is not None:
                # init feats container
                curr_upos.setdefault("vals", {})
                feats = curr_upos["vals"]

                for f in w.feats.split("|"):
                    k, v = f.split("=")

                    # init feat
                    feat = feats.setdefault(k, {"count": 0, "vals": {}})
                    feat["count"] += 1

                    # init and count value
                    feat["vals"].setdefault(v, 0)
                    feat["vals"][v] += 1

        # print('---')
        # print()
        # print('upos')
        # print()
        # stats_upos = defaultdict(   # upos
        #     lambda: defaultdict(    # feats
        #         int                 # count
        #     )
        # )
        # counts_upos = defaultdict(int)
        # for w in all_words:
        #     feats = (w.feats or 'no_features').split('|')
        #     counts_upos[w.upos] += 1
        #     for f in feats:
        #         stats_upos[w.upos][f] += 1
        #
        # for key, value in sorted(stats_upos.items()):
        #     count = counts_upos[key]
        #     assert count == upos_dict["vals"][key]["count"]
        #     print(key, f'({count}, {count / all_words_count * 100:.2f}%)')
        #     if isinstance(value, defaultdict):
        #         for k, v in sorted(value.items()):
        #             if k != "no_features":
        #                 foo, bar = k.split("=")
        #                 assert v == upos_dict["vals"][key]["vals"][foo]["vals"][bar]
        #             print(f'    {k} ({v}, {v / count * 100:.2f}%)')
        #     else:
        #         print(f'    {key}: {value}')

    with Timer(name='Dependency Relations', text=timer_text):
        deprel_dict = {}
        for w in all_words:
            key = w.deprel
            deprel_dict.setdefault(key, 0)
            deprel_dict[key] += 1

        # print('---')
        # print()
        # print('deprel:')
        # print()
        #
        # stats_deprel = defaultdict(int)
        # for w in all_words:
        #     stats_deprel[w.deprel] += 1
        # for k, v in sorted(stats_deprel.items()):
        #     print(f'{k} = {v}')
        # for w in all_words:
        #     assert deprel[w.deprel] == stats_deprel[w.deprel]

    return {
        "upos": upos_dict,
        "deprel": deprel_dict
    }
