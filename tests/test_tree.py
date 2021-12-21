import unittest
from graded_readers_stats.tree import *
import pandas as pd
from anytree import Node, LevelOrderGroupIter


class TreeTestCase(unittest.TestCase):
    def test_tree_props(self):
        # Given
        column = 'Reader_Avanzado_trees'
        data = {
            column: [[[
                self.make_tree(1, 3),  # min_width: 1, min_height: 3
                self.make_tree(2, 4),  # avg_width: 2, avg_height: 4
                self.make_tree(3, 5),  # max_width: 3, max_height: 5
            ]]],
        }
        df = pd.DataFrame(data)

        # When
        new_column, tree_props = calculate_tree_props(df, column)

        # Then
        self.assertEqual(column + TREES_PROPS, new_column)
        self.assertEqual((1, 3, 2, 3, 5, 4), tree_props[0])

    @staticmethod
    def make_tree(width, height):
        assert width > 0
        assert height > 0

        height_origin = height + 1
        root = Node("root")
        head = root

        while height > 0:
            node = Node("level_" + str(height_origin - height))
            node.parent = head
            head = node
            height -= 1

        width -= 1  # take into account height node
        while width > 0:
            node = Node("level_1_" + str(width))
            node.parent = root
            width -= 1

        return root


if __name__ == '__main__':
    unittest.main()
