import pytest

from graded_readers_stats.tree import *


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@pytest.fixture
def storage():
    storage = {
        'tree': {},
        'stanza': [  # list of docs
            DotDict({  # a doc
                'sentences': [

                    # Head
                    #   > Word1
                    #       > Word3
                    #   > Word2
                    #       > Word4
                    #       > Word5
                    DotDict({'words': [  # sentence a.k.a. a sequence of words
                        DotDict({
                            'id': 1,
                            'lemma': 'Head',
                            'head': 0
                        }),
                        DotDict({
                            'id': 2,
                            'lemma': 'Word1',
                            'head': 1
                        }),
                        DotDict({
                            'id': 4,
                            'lemma': 'Word3',
                            'head': 2
                        }),
                        DotDict({
                            'id': 3,
                            'lemma': 'Word2',
                            'head': 1
                        }),
                        DotDict({
                            'id': 5,
                            'lemma': 'Word4',
                            'head': 3
                        }),
                        DotDict({
                            'id': 6,
                            'lemma': 'Word5',
                            'head': 3
                        }),
                    ]}),

                    # Head
                    #   > Word1
                    #       > Word3
                    #       > Word6 NEW
                    #   > Word2
                    #       > Word4
                    #       > Word5
                    #          > Word7 NEW
                    DotDict({'words': [
                        DotDict({
                            'id': 1,
                            'lemma': 'Head',
                            'head': 0
                        }),
                        DotDict({
                            'id': 2,
                            'lemma': 'Word1',
                            'head': 1
                        }),
                        DotDict({
                            'id': 4,
                            'lemma': 'Word3',
                            'head': 2
                        }),
                        DotDict({
                            'id': 7,
                            'lemma': 'Word6',
                            'head': 2
                        }),
                        DotDict({
                            'id': 3,
                            'lemma': 'Word2',
                            'head': 1
                        }),
                        DotDict({
                            'id': 5,
                            'lemma': 'Word4',
                            'head': 3
                        }),
                        DotDict({
                            'id': 6,
                            'lemma': 'Word5',
                            'head': 3
                        }),
                        DotDict({
                            'id': 8,
                            'lemma': 'Word7',
                            'head': 6
                        }),
                    ]})
                ]
            })
        ]
    }
    return DotDict(storage)


def test_make_tree_for_loc(storage):
    # When
    tree = make_tree_for_loc(storage, 0, 0)

    # Then
    assert "Node('/Head')" == repr(tree)
    assert "(Node('/Head/Word1'), Node('/Head/Word2'))" == repr(tree.children)
    assert "(Node('/Head/Word1/Word3'), "\
           "Node('/Head/Word2/Word4'), "\
           "Node('/Head/Word2/Word5'))" == repr(tree.leaves)


def test_get_tree_props(storage):
    # Given
    docs_trees = [[
        make_tree_for_loc(storage, 0, 0),
        make_tree_for_loc(storage, 0, 1),
    ]]

    # When
    props = get_tree_props(docs_trees)

    # Then
    assert (3, 4, 3.5, 2, 3, 2.5) == props
