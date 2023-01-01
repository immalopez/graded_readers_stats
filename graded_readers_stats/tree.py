##############################################################################
#                               TREE CALCULATIONS                            #
##############################################################################

import funcy
from anytree import Node, LevelOrderGroupIter
from stanza.models.common.doc import Sentence

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def make_trees_for_terms_docs_sents_term_loc(storage, terms_locs):
    return [make_tree_for_docs_sents_term_loc(storage, docs_sents_term_loc)
            for docs_sents_term_loc in terms_locs]


def make_tree_for_docs_sents_term_loc(storage, docs_sents_term_loc):
    return [[make_tree_for_loc(storage, doc_index, term_loc[0])  # 0 = sent idx
             for term_loc in sents_locs]
            for doc_index, sents_locs in enumerate(docs_sents_term_loc)]


def make_trees_for_docs_terms_sents_term_loc(storage, docs_locs):
    return [
        make_tree_for_terms_sents_term_loc(
            storage,
            doc_idx,
            terms_sents_term_loc
        )
        for doc_idx, terms_sents_term_loc in enumerate(docs_locs)
    ]


def make_tree_for_terms_sents_term_loc(
        storage,
        doc_index,
        terms_sents_term_loc
):
    return [[make_tree_for_loc(storage, doc_index, term_loc[0])  # 0 = sent idx
             for term_loc in sents_locs]
            for sents_locs in terms_sents_term_loc]


def make_tree_for_loc(storage, doc_idx, sent_idx) -> Node:
    stanza_docs = storage['stanza']
    sent = stanza_docs[doc_idx].sentences[sent_idx]
    trees = storage['tree']
    key = f'{doc_idx}_{sent_idx}'

    if key in trees:
        return trees[key]

    nodes = {word.id: Node(word.lemma) for word in sent.words}
    for word in sent.words:
        nodes[word.id].parent = nodes[word.head] if word.head > 0 else None
        if word.head == 0:
            root = nodes[word.id]

    storage['tree'][key] = root
    return root


def calculate_tree_props_v2(terms_trees):
    return [get_tree_props(docs_trees) for docs_trees in terms_trees]


def get_tree_props(
        docs_trees: [[Node]]
) -> (int, int, int, int, int, int):
    # min_width, max_width, avg_width, min_height, max_height, avg_height

    min_w, min_h = 1000, 1000
    max_w, max_h = 0, 0
    sum_w, sum_h = 0, 0
    num_nodes = 0

    for doc_trees in docs_trees:
        for sent_tree in doc_trees:
            width, height = get_tree_props_for_sent(sent_tree)

            sum_w += width
            sum_h += height
            num_nodes += 1

            min_w = min(min_w, width)
            max_w = max(max_w, width)
            min_h = min(min_h, height)
            max_h = max(max_h, height)

    avg_w = sum_w / num_nodes if num_nodes > 0 else None
    avg_h = sum_h / num_nodes if num_nodes > 0 else None

    has_result = max_w > 0
    if has_result:
        return min_w, max_w, avg_w, min_h, max_h, avg_h
    else:
        return None, None, None, None, None, None


def get_tree_props_for_sent(sent_tree):
    width = max([len(children)
                 for children in LevelOrderGroupIter(sent_tree)])
    height = sent_tree.height
    return width, height


##############################################################################
#                                Deprecated                                  #
##############################################################################


def deprecated_make_trees_for_occurrences_v2(
        terms_locs: [],
        stanza_docs: Series
):
    return [deprecated_make_tree(term_locs, stanza_docs)
            for term_locs in terms_locs]


def deprecated_make_tree(
        docs_locs: [[(int, (int, int))]],  # list of docs of sents
        stanza_docs: Series
) -> [(int, Node)]:
    docs_trees = []
    for doc_index, doc_locs in enumerate(docs_locs):
        doc_trees = []
        for sent_loc in doc_locs:
            sent_index = sent_loc[0]
            sent = stanza_docs[doc_index].sentences[sent_index]
            root_node = deprecated_make_tree_for_sent(sent)
            doc_trees.append(root_node)
        docs_trees.append(doc_trees)
    return docs_trees


def deprecated_make_tree_for_sent(sent: Sentence) -> Node:
    nodes = {word.id: Node(word.lemma) for word in sent.words}
    for word in sent.words:
        nodes[word.id].parent = nodes[word.head] if word.head > 0 else None
        if word.head == 0:
            root = nodes[word.id]
    return root


def deprecated_make_trees_for_occurrences(
        vocab: DataFrame,
        texts: DataFrame,
        column: str
) -> (str, [[(int, Node)]]):
    in_column_locations = column + ' ' + LOCATIONS
    out_column_trees = column + ' ' + 'trees'

    rows = vocab[in_column_locations]
    trees = [deprecated_make_tree(row, texts[COL_STANZA_DOC]) for row in rows]
    return out_column_trees, trees


def deprecated_calculate_tree_props(
        vocabs: DataFrame,
        column: str
) -> (str, Series):
    in_column_trees = column
    out_column_trees_props = column + TREES_PROPS
    values = vocabs.apply(
        lambda x: get_tree_props(x[in_column_trees]),
        axis=1
    )
    return out_column_trees_props, values


##############################################################################
#                                 PIPELINE                                   #
##############################################################################


terms_tree_props_pipeline = funcy.rcompose(
    make_trees_for_terms_docs_sents_term_loc,
    calculate_tree_props_v2
)
texts_tree_props_pipeline = funcy.rcompose(
    make_trees_for_docs_terms_sents_term_loc,
    calculate_tree_props_v2
)
