##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from anytree import Node, RenderTree, LevelOrderGroupIter
from graded_readers_stats.constants import *
from graded_readers_stats.utils import *
from graded_readers_stats._typing import *


def get_tree_widths_and_depths(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy
) -> None:
    for group_name in sentences_by_groups.groups:

        phrase_series = [wrapper[0] for wrapper in phrases[COL_LEMMA]]
        widths = []
        heights = []
        for phrase in phrase_series:
            max_width = 0
            max_height = 0

            docs = sentences_by_groups.get_group(group_name)[COL_STANZA_DOC]
            for doc in docs:

                sents_lemma = doc.get('lemma', True)
                for (sent, sent_lemma) in zip(doc.sentences, sents_lemma):

                    if get_range_of_phrase_in_sentence(phrase, sent_lemma):

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
                        # for pre, fill, node in RenderTree(root):
                        #     print("%s%s" % (pre, node.name))

            widths.append(max_width)
            heights.append(max_height)

        phrases[group_name + SUFFIX_WIDTHS] = widths
        phrases[group_name + SUFFIX_HEIGHTS] = heights
