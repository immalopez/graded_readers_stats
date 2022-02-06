from typing import List, Set


def collect_context_words_multiple(terms_locs, docs, window) -> List[Set[str]]:
    return [collect_context_words_single(term_locs, docs, window)
            for term_locs in terms_locs]


def collect_context_words_single(term_locs, docs, window) -> Set[str]:
    from string import punctuation

    term_context = []
    # for every doc
    for doc_index, doc_locs in enumerate(term_locs):
        doc = docs[doc_index]
        # for every sentence
        for sent_loc in doc_locs:
            sent_index = sent_loc[0]
            start, end = sent_loc[1]
            sent = doc[sent_index]

            # normalize and remove irrelevant context words
            sent_lemmas = [word.lower()
                           for word in sent
                           if word not in punctuation + r'¡¿–—']

            # limit indices to sentence bounds using `min` and `max`
            # to safely use with list slicing
            # since out-of-bounds slicing would return empty list ([])
            slice_before = slice(max(0, start - window), start)
            slice_after = slice(end, min(end + window, len(sent_lemmas)))
            context = sent_lemmas[slice_before] + sent_lemmas[slice_after]
            term_context.append(context)

    # flatten term_context since we appended lists of words to it
    flattened = [word for words in term_context for word in words]

    # deduplicated
    return set(flattened)
