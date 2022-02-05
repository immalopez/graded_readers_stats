from graded_readers_stats import preprocess
import pytest


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
    term_locs = preprocess.find_term_locations_in_docs(terms, docs)

    # Then
    assert len(terms) == len(term_locs)
    assert len(docs) == len(term_locs[0])
    assert len(docs) == len(term_locs[1])


def test_locations_details(terms, docs):
    # When
    term_locs = preprocess.find_term_locations_in_docs(terms, docs)

    # Then
    expected = [
        [[(0, (0, 2))], [], []],
        [[], [], [(1, (6, 7))]]
    ]
    assert expected == term_locs
