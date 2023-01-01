from statistics import mean
from typing import List, Set

from funcy import partial, rpartial, rcompose
from pandas.core.common import flatten

from graded_readers_stats.frequency import freqs_by_term, count_terms
from graded_readers_stats.preprocess import locate_terms_in_docs
from graded_readers_stats.tfidf import tfidfs
from graded_readers_stats.tree import terms_tree_props_pipeline


def collect_context_words_by_terms(terms_locs, docs, window) -> List[Set[str]]:
    return [collect_context_words_for_term(term_locs, docs, window)
            for term_locs in terms_locs]


def collect_context_words_for_term(term_locs, docs, window) -> Set[str]:
    from string import punctuation

    contexts = []
    for doc_index, doc_locs in enumerate(term_locs):
        doc = docs[doc_index]
        contexts.append(
            context_words_for_doc_and_sent_idx(doc, doc_locs, window)
        )

    flattened = [word for words in contexts for word in words]
    sanitized = [word.lower()
                 for word in flattened
                 if word not in punctuation + r'¡¿–—']

    return set(sanitized)


def collect_context_words_by_docs(docs_locs, docs, window) -> List[Set[str]]:
    return [
        collect_context_words_for_sents_locs(doc_idx, sents_locs, docs, window)
        for doc_idx, sents_locs in enumerate(docs_locs)
    ]


def collect_context_words_for_sents_locs(doc_idx, sents_locs, docs, window) -> Set[str]:
    from string import punctuation

    contexts = []
    for term_locs in sents_locs:
        doc = docs[doc_idx]
        contexts.append(
            context_words_for_doc_and_sent_idx(doc, term_locs, window)
        )

    flattened = [word for words in contexts for word in words]
    sanitized = [word.lower()
                 for word in flattened
                 if word not in punctuation + r'¡¿–—']

    return set(sanitized)


def context_words_for_doc_and_sent_idx(doc, locs, window):
    contexts = []
    for sent_loc in locs:
        sent_index = sent_loc[0]
        start, end = sent_loc[1]
        sent = doc[sent_index]

        # limit indices to sentence bounds using `min` and `max`
        # to safely use with list slicing
        # since out-of-bounds slicing would return empty list ([])
        slice_before = slice(max(0, start - window), start)
        slice_after = slice(end, min(end + window, len(sent)))
        context = sent[slice_before] + sent[slice_after]
        contexts.append(context)

    # flatten
    return [word for context in contexts for word in context]


def locate_ctx_terms_in_docs(ctx_words_by_term, texts):
    ctx_terms_flat = list(flatten(ctx_words_by_term))
    ctx_terms_wrap = [[word] for word in ctx_terms_flat]
    ctx_terms_locs = locate_terms_in_docs(ctx_terms_wrap, texts)
    ctx_term_loc_dict = dict(zip(ctx_terms_flat, ctx_terms_locs))
    ctxs_locs = ctxs_locs_by_term(ctx_term_loc_dict, ctx_words_by_term)
    return ctxs_locs


def ctxs_locs_by_term(term_loc_dict, ctx_words_by_term):
    return [[term_loc_dict[w] for w in ws]
            for ws in ctx_words_by_term]


def count_pipeline():
    return rcompose(
        partial(map, count_terms),
        # partial(map, avg),
    )


def freqs_pipeline(words_total):
    return rcompose(
        partial(map, rpartial(freqs_by_term, words_total)),
        partial(map, avg),
    )


def tfidfs_pipeline(texts):
    return rcompose(
        partial(map, rpartial(tfidfs, texts)),
        partial(map, avg)
    )


def trees_pipeline(storage):
    empty_tuple = (None, None, None, None, None, None)
    return rcompose(
        partial(map, partial(terms_tree_props_pipeline, storage)),
        partial(map, rpartial(avg_tuples, empty_tuple))
    )


def avg(data):
    return mean(data) if len(data) > 0 else 0


def avg_tuples(data, empty_value):
    if not data:
        return empty_value
    transposed = list(map(list, zip(*data)))
    return tuple(map(lambda d: mean(d) if len(d) > 0 else 0, transposed))

