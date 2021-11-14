##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from anytree import Node, LevelOrderGroupIter
from stanza.models.common.doc import Sentence

from graded_readers_stats.constants import *
from graded_readers_stats._typing import *


def make_trees_for_occurrences(
        vocabs: DataFrame,
        texts: DataFrame,
        column: str
) -> None:
    in_column_locations = column + ' ' + LOCATIONS
    out_column_trees = column + ' ' + 'trees'

    rows = vocabs[in_column_locations]
    trees = [make_tree(row, texts[COL_STANZA_DOC]) for row in rows]
    vocabs[out_column_trees] = trees


def make_tree(
        doc_locations: [[(int, (int, int))]],  # list of docs of sents
        docs: Series
) -> [(int, Node)]:
    doc_trees = []
    for doc_index, doc_location in enumerate(doc_locations):
        sent_trees = []
        if len(doc_location) > 0:
            for sent_location in doc_location:
                sent_index = sent_location[0]
                sent = docs[doc_index].sentences[sent_index]
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
) -> None:
    in_column_trees = column + ' ' + TREES
    out_column_trees_props = column + ' ' + TREES_PROPS
    vocabs[out_column_trees_props] = vocabs.apply(
        lambda x: get_tree_props(x[in_column_trees]),
        axis=1
    )


def get_tree_props(
        doc_trees: [[Node]]
) -> (int, int, int, int):  # min_width, max_width, min_height, max_height

    from collections import namedtuple
    TreeProps = namedtuple(
        'TreeProps',
        ['min_width', 'max_width', 'min_height', 'max_height']
    )

    min_w, max_w, min_h, max_h = None, None, None, None

    for doc in doc_trees:
        for node in doc:

            width = max([len(children)
                         for children in LevelOrderGroupIter(node)])
            height = node.height

            min_w = width if min_w is None else min(min_w, width)
            max_w = width if max_w is None else max(max_w, width)
            min_h = height if min_h is None else min(min_h, height)
            max_h = height if max_h is None else max(max_h, height)

    return TreeProps(min_w, max_w, min_h, max_h)
