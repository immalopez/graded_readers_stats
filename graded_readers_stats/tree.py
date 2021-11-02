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
                        # print(max_width_sent, max_height_sent)

                        max_width = max(max_width, max_width_sent)
                        max_height = max(max_height, max_height_sent)

                        # print tree
                        # for pre, fill, node in RenderTree(root):
                        #     print("%s%s" % (pre, node.name))

            widths.append(max_width)
            heights.append(max_height)

        phrases[group_name + SUFFIX_WIDTHS] = widths
        phrases[group_name + SUFFIX_HEIGHTS] = heights


##############################################################################
#                                   EXAMPLES                                 #
##############################################################################

# doc = nlp_es("A Plamen le gustan las motos.")
#
# root = None
# for sent in doc.sentences:
#     nodes = {word.id: Node(word.lemma) for word in sent.words}
#     for word in sent.words:
#         nodes[word.id].parent = nodes[word.head] if word.head > 0 else None
#         if word.head == 0:
#             root = nodes[word.id]
# print(max(len(node.siblings) for node in root.descendants))
# print([node.name for node in root.descendants])
# for _, _, node in RenderTree(root):
#     print(node.name, len(node.siblings))
#
# for pre, fill, node in RenderTree(root):
#     print("%s%s" % (pre, node.name))
#


# root = Node(10)
#
# level_1_child_1 = Node(34, parent=root)
# level_1_child_2 = Node(89, parent=root)
# level_2_child_1 = Node(45, parent=level_1_child_1)
# level_2_child_2 = Node(50, parent=level_1_child_2)
#
# for pre, fill, node in RenderTree(root):
#     print("%s%s" % (pre, node.name))


# import stanza
#
# nlp = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma,depparse')
# doc = nlp('Nous avons atteint la fin du sentier.')
# print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


