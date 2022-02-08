from graded_readers_stats.tree import *
import pandas as pd
from anytree import Node, LevelOrderGroupIter


def test_tree_props():
    # Given
    column = 'Reader_Avanzado_trees'
    data = {
        column: [[[
            make_tree(1, 3),
            make_tree(2, 4),
            make_tree(3, 5),
        ]]],
    }
    df = pd.DataFrame(data)

    # When
    new_column, tree_props = calculate_tree_props(df, column)

    # Then
    assert column + TREES_PROPS == new_column
    assert (1, 3, 2, 3, 5, 4) == tree_props[0]


def test_calculate_tree_props_v2():
    # Given
    trees = [
        [[make_tree(1, 3), make_tree(2, 4), make_tree(3, 5)]],
        [[make_tree(2, 2), make_tree(4, 4), make_tree(6, 6)]],
        #          ^ low            ^ avg            ^ high
    ]

    # When
    tree_props = calculate_tree_props_v2(trees)

    # Then
    # min_w, max_w, avg_w, min_h, max_h, avg_h
    expected = [
        (1, 3, 2, 3, 5, 4),
        (2, 6, 4, 2, 6, 4),
    ]
    assert expected == tree_props


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
