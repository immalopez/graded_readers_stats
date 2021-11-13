##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from anytree import Node, LevelOrderGroupIter
from stanza.models.common.doc import Sentence

from graded_readers_stats.constants import *
from graded_readers_stats.utils import *
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
        nodes[word.id].parent = nodes[word.head] \
            if word.head > 0 \
            else None
        if word.head == 0:
            root = nodes[word.id]
    return root

def make_trees_for_locations(
        locations: [[(int, (int, int))]],
        docs: Series
) -> [[(int, Node)]]:
    # trees_by_doc = []
    # for sent_loc in doc_locations:

    for doc_index, doc in docs.items():
        sent_locs = locations[doc_index]
        for loc in sent_locs:
            sent = doc.sentences[loc[0]]
            nodes = {word.id: Node(word.lemma) for word in sent.words}
            for word in sent.words:
                nodes[word.id].parent = nodes[word.head] \
                    if word.head > 0 \
                    else None
                if word.head == 0:
                    root = nodes[word.id]



def get_tree_widths_and_depths_v0(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy
) -> None:
    for group_name in sentences_by_groups.groups:

        vocab_lemmas = [wrapper[0] for wrapper in vocabs[COL_LEMMA]]
        widths = []
        heights = []
        for vocab in vocab_lemmas:
            max_width = 0
            max_height = 0

            docs = sentences_by_groups.get_group(group_name)[COL_STANZA_DOC]
            for doc in docs:

                sents_lemma = doc.get('lemma', True)
                for (sent, sent_lemma) in zip(doc.sentences, sents_lemma):

                    if first_occurrence_of_vocab_in_sentence(vocab, sent_lemma):

                        nodes = {word.id: Node(word.lemma) for word in sent.words}
                        for word in sent.words:
                            nodes[word.id].parent = nodes[word.head] \
                                if word.head > 0 \
                                else None
                            if word.head == 0:
                                root = nodes[word.id]

                        max_height_sent = root.height
                        max_width_sent = max([
                            len(children)
                            for children in LevelOrderGroupIter(root)
                        ])

                        max_width = max(max_width, max_width_sent)
                        max_height = max(max_height, max_height_sent)

                        # print tree
                        # from anytree import RenderTree
                        # for pre, fill, node in RenderTree(root):
                        #     print("%s%s" % (pre, node.name))

            widths.append(max_width)
            heights.append(max_height)

        vocabs[group_name + ' ' + WIDTHS] = widths
        vocabs[group_name + ' ' + HEIGHTS] = heights


def calculate_tree_props(
        vocabs: DataFrame,
        column: str
) -> None:
    in_column_trees = column + ' ' + TREES
    out_column_trees_props = column + ' ' + TREES_PROPS

    trees = vocabs[in_column_trees]
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
