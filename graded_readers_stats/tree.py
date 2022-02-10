##############################################################################
#                               TREE CALCULATIONS                            #
##############################################################################

import funcy
from anytree import Node, LevelOrderGroupIter
from stanza.models.common.doc import Sentence

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def make_trees_for_terms_locs(storage, terms_locs):
    return [make_tree_for_term_locs(storage, term_locs)
            for term_locs in terms_locs]


def make_tree_for_term_locs(storage, docs_locs):
    return [make_tree_for_loc(storage, doc_index, sent_loc[0])
            for doc_index, doc_locs in enumerate(docs_locs)
            for sent_loc in doc_locs]


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
    return [get_tree_props(term_trees) for term_trees in terms_trees]


def get_tree_props(
        docs_trees: [[Node]]
) -> (int, int, int, int):  # min_width, max_width, min_height, max_height

    min_w, min_h = 1000, 1000
    max_w, max_h = 0, 0
    sum_w, sum_h = 0, 0
    num_nodes = 0

    for doc_trees in docs_trees:
        for sent_tree in doc_trees:

            width = max([len(children)
                         for children in LevelOrderGroupIter(sent_tree)])
            height = sent_tree.height

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
    return None, None, None, None, None, None

##############################################################################
#                                Deprecated                                  #
##############################################################################


def make_trees_for_occurrences_v2(terms_locs: [], stanza_docs: Series):
    return [make_tree(term_locs, stanza_docs) for term_locs in terms_locs]


def make_tree(
        docs_locs: [[(int, (int, int))]],  # list of docs of sents
        stanza_docs: Series
) -> [(int, Node)]:
    docs_trees = []
    for doc_index, doc_locs in enumerate(docs_locs):
        doc_trees = []
        for sent_loc in doc_locs:
            sent_index = sent_loc[0]
            sent = stanza_docs[doc_index].sentences[sent_index]
            root_node = make_tree_for_sent(sent)
            doc_trees.append(root_node)
        docs_trees.append(doc_trees)
    return docs_trees


def make_tree_for_sent(sent: Sentence) -> Node:
    nodes = {word.id: Node(word.lemma) for word in sent.words}
    for word in sent.words:
        nodes[word.id].parent = nodes[word.head] if word.head > 0 else None
        if word.head == 0:
            root = nodes[word.id]
    return root


def make_trees_for_occurrences(
        vocab: DataFrame,
        texts: DataFrame,
        column: str
) -> (str, [[(int, Node)]]):
    in_column_locations = column + ' ' + LOCATIONS
    out_column_trees = column + ' ' + 'trees'

    rows = vocab[in_column_locations]
    trees = [make_tree(row, texts[COL_STANZA_DOC]) for row in rows]
    return out_column_trees, trees


def calculate_tree_props(
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


tree_props_pipeline = funcy.rcompose(
    make_trees_for_terms_locs,
    calculate_tree_props_v2
)
