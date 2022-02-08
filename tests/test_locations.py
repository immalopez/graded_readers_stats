import pytest

from graded_readers_stats import preprocess
from graded_readers_stats.context import locate_ctx_terms_in_docs


@pytest.fixture
def terms():
    # list of lists because some terms are multi-word
    return [['this', 'is'], ['world']]


@pytest.fixture
def docs():
    # docs are list of sentences which are list of words
    return [
        [['this', 'is', 'a', 'sentence', '.'], ['this', 'as', 'well']],
        [['hello', 'good', 'sir']],
        [['yes'], ['you', 'are', 'the', 'best', 'in', 'the', 'world', '!']]
    ]


def test_locations_structure(terms, docs):
    # When
    term_locs = preprocess.locate_terms_in_docs(terms, docs)

    # Then
    assert len(terms) == len(term_locs)
    assert len(docs) == len(term_locs[0])
    assert len(docs) == len(term_locs[1])


def test_locations_details(terms, docs):
    # When
    term_locs = preprocess.locate_terms_in_docs(terms, docs)

    # Then
    expected = [
        [[(0, (0, 2))], [], []],
        [[], [], [(1, (6, 7))]]
    ]
    assert expected == term_locs


def test_context_locations():
    # Given
    ctx_words = [
        {'hello', 'world'},
        {'good'},
        {}
    ]
    docs = [
        [['this', 'is', 'a', 'sentence', '.'], ['this', 'as', 'well']],
        [['hello', 'good', 'sir']],
        [['yes'], ['you', 'are', 'the', 'best', 'in', 'the', 'world', '!']]
    ]

    # When
    locations = locate_ctx_terms_in_docs(ctx_words, docs)

    # Then
    expected = [  # list of terms
        [  # term 1
            [  # hello
                [],             # doc 1
                [(0, (0, 1))],  # doc 2
                []              # doc 3
            ], [  # world
                [],
                [],
                [(1, (6, 7))]
            ]
        ],
        [  # term 2
            [  # good
                [],
                [(0, (1, 2))],
                []
            ]
        ],
        []  # term 3
    ]
    assert expected == locations

