from statistics import mean
from typing import List, Set


def collect_context_words(terms_locs, docs, window) -> List[Set[str]]:
    return [collect_context_words_single(term_locs, docs, window)
            for term_locs in terms_locs]


def collect_context_words_single(term_locs, docs, window) -> Set[str]:
    from string import punctuation

    contexts = []
    # for every doc
    for doc_index, doc_locs in enumerate(term_locs):
        doc = docs[doc_index]
        # for every sentence
        for sent_loc in doc_locs:
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

    # flatten contexts since we accumulate lists, not plain words
    flattened = [word for words in contexts for word in words]

    # normalize and remove irrelevant context words
    sanitized = [word.lower()
                 for word in flattened
                 if word not in punctuation + r'¡¿–—']

    # deduplicated
    return set(sanitized)


def ctxs_locs_by_term(term_loc_dict, ctx_words_by_term):
    return [[term_loc_dict[w] for w in ws]
            for ws in ctx_words_by_term]


def avg(data):
    return mean(data) if len(data) > 0 else 0


def avg_tuples(data, empty_value):
    if not data:
        return empty_value

    transposed = list(map(list, zip(*data)))
    return tuple(map(lambda d: mean(d) if len(d) > 0 else 0, transposed))

