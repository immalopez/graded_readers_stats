##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

import funcy
from anytree import Node, LevelOrderGroupIter
from stanza.models.common.doc import Sentence

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


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


def make_trees_for_occurrences_v2(
        terms_locs: [],
        stanza_docs: Series
) -> (str, [[(int, Node)]]):
    return [make_tree(term_locs, stanza_docs) for term_locs in terms_locs]


def make_tree(
        docs_locs: [[(int, (int, int))]],  # list of docs of sents
        stanza_docs: Series
) -> [(int, Node)]:
    doc_trees = []
    for doc_index, doc_locs in enumerate(docs_locs):
        sent_trees = []
        if len(doc_locs) > 0:
            for sent_location in doc_locs:
                sent_index = sent_location[0]
                sent = stanza_docs[doc_index].sentences[sent_index]
                root_node = make_tree_for_sent(sent)
                sent_trees.append(root_node)
        doc_trees.append(sent_trees)
    return doc_trees


def make_tree_for_sent(sent: Sentence) -> Node:
    nodes = {word.id: Node(word.lemma) for word in sent.words}
    for word in sent.words:
        nodes[word.id].parent = nodes[word.head] if word.head > 0 \
            else None
        if word.head == 0:
            root = nodes[word.id]
    return root


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


def calculate_tree_props_v2(terms_trees):
    return [get_tree_props(term_trees) for term_trees in terms_trees]


def get_tree_props(
        docs_trees: [[Node]]
) -> (int, int, int, int):  # min_width, max_width, min_height, max_height

    min_w, max_w = None, None
    min_h, max_h = None, None
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

            min_w = width if min_w is None else min(min_w, width)
            max_w = width if max_w is None else max(max_w, width)
            min_h = height if min_h is None else min(min_h, height)
            max_h = height if max_h is None else max(max_h, height)

    avg_w = sum_w / num_nodes if num_nodes > 0 else None
    avg_h = sum_h / num_nodes if num_nodes > 0 else None
    return min_w, max_w, avg_w, min_h, max_h, avg_h


tree_props_pipeline = funcy.rcompose(
    make_trees_for_occurrences_v2,
    calculate_tree_props_v2
)
