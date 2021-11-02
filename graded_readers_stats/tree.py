##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from anytree import Node, RenderTree
from graded_readers_stats.preprocess import *


def find_min_max_width(vocabulary, readers_by_level, column_prefix):
    return None


def find_min_max_depth(vocabulary, readers_by_level, column_prefix):
    return None


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


