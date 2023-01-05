from __future__ import annotations

import pandas as pd
from codetiming import Timer
from lexicalrichness import LexicalRichness

from graded_readers_stats.constants import (
    COL_STANZA_DOC,
)
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline,
)


def get_lexical_richness(text):
    """
    Calculates the lexical richness of the text.
    """
    lex = LexicalRichness(text)
    return {
        "msttr": lex.msttr(segment_window=100),
        "mattr": lex.mattr(window_size=100),
        "ttr": lex.ttr,
        "rttr": lex.rttr,
        "cttr": lex.cttr,
        "mtld": lex.mtld(threshold=0.72),
        "hdd": lex.hdd(draws=42),
        "Herdan": lex.Herdan,
        "Summer": lex.Summer,
        "Dugast": lex.Dugast,
        "Maas": lex.Maas,
    }


def calc_stats_for_group(
        group_name: str,
        group_df: pd.DataFrame,
        max_docs: int
):
    timer_text = '{name}: {:0.0f} seconds'
    texts_df = group_df

    if max_docs:
        texts_df = texts_df[:max_docs].copy()

    with Timer(name='Preprocess', text=timer_text):
        texts_df = run(texts_df, text_analysis_pipeline)

    with Timer(name='UPOS', text=timer_text):
        stanza_docs = texts_df[COL_STANZA_DOC]
        stats_per_doc = [calc_stats_for_stanza_doc(doc) for doc in stanza_docs]
        return stats_per_doc


def calc_stats_for_stanza_doc(doc):
    all_words = [word
                 for sent in doc.sentences
                 for word in sent.words]
    all_words_count = len(all_words)

    upos_dict = {"count": 0, "vals": {}}
    for w in all_words:
        # count general words
        upos_dict["count"] += 1

        # init and count the current upos
        curr_upos = upos_dict["vals"].setdefault(w.upos, {"count": 0})
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

    deprel_dict = {}
    for w in all_words:
        key = w.deprel
        deprel_dict.setdefault(key, 0)
        deprel_dict[key] += 1

    return {
        "upos": upos_dict,
        "deprel": deprel_dict
    }


def find(data, keys):
    rv = data
    for key in keys:
        rv = rv.get(key, {})
    return rv if rv != {} else 0


def make_key_str(keys):
    return "-".join(
        [key for key in keys
         if key not in ["vals", "count"]]
    )


def group_upos_values_by_key(result, next, docs, path):
    for k, v in next.items():
        cur_path = path + [k]
        key = make_key_str(cur_path)
        if isinstance(v, dict):
            group_upos_values_by_key(result, v, docs, cur_path)
        else:
            result[key] = [find(doc, cur_path) for doc in docs]
