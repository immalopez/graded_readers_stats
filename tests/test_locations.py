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


def test_transpose_locations_from_terms_to_docs_with_word_el_at_index_0():
    terms = [
        ["agua"],
        ["coche"],
    ]
    docs = [
        [  # doc 1
            ["el", "agua", "fría", "."],
            ["esta", "frase", "vacía", "."],
            ["el", "agua", "caliente", "."],
        ],
        [  # doc 2
            ["el", "coche", "."],
        ],
        [  # doc 3
            ["empty"]
        ]
    ]
    terms_count = len(terms)
    docs_count = len(docs)
    ctx_words_by_terms = [
        {'el', 'caliente', 'fría'},  # term "agua"
        {'el'},                      # term "coche"
    ]
    ctxs_locs_by_terms = \
        [
            [  # term "agua"
                [  # context "el"
                    [(0, (0, 1)), (2, (0, 1))],  # doc
                    [(0, (0, 1))],               # doc
                    [],                          # doc
                ],
                [  # context "caliente"
                    [(2, (2, 3))],  # doc
                    [],             # doc
                    [],             # doc
                ],
                [  # context "fría"
                    [(0, (2, 3))],  # doc
                    [],             # doc
                    [],             # doc
                ]
            ],
            [  # term "coche"
                [  # context "el"
                    [(0, (0, 1)), (2, (0, 1))],  # doc
                    [(0, (0, 1))],               # doc
                    [],                          # doc
                ]
            ]
        ]

    # When
    docs_locs = transpose_ctx_terms_to_docs_locations(
        ctxs_locs_by_terms,
        terms_count,
        docs_count,
    )

    # Then
    assert len(docs_locs) == docs_count, "docs count"
    assert len(docs_locs[0]) == 3, "context words count"
    assert len(docs_locs[0][0]) == terms_count, "terms count per context"
    assert len(docs_locs[0][0][0]) == 2, "'el' appears twice in first doc"
    assert len(docs_locs[0][0][1]) == 2, "'el' appears twice in first doc"
    assert len(docs_locs[1][0][0]) == 1, "'el' appears once in second doc"
    assert [
        [  # doc
            [  # context "el"
                [(0, (0, 1)), (2, (0, 1))],  # term "agua"
                [(0, (0, 1)), (2, (0, 1))],  # term "coche"
            ],
            [  # context "caliente"
                [(2, (2, 3))],  # term "agua"
                []              # term "coche"
            ],
            [  # context "fría"
                [(0, (2, 3))],  # term "agua"
                []              # term "coche"
            ]
        ],
        [  # doc
            [  # context "el"
                [(0, (0, 1))],  # term "agua"
                [(0, (0, 1))]   # term "coche"
            ]
        ],
        [  # doc
            # no context
        ]
    ] == docs_locs



def transpose_ctx_terms_to_docs_locations(
        ctxs_locs_by_terms,
        terms_count,
        docs_count,
):
    ctxs_locs_by_docs = [[] for _ in range(docs_count)]
    for ti, t in enumerate(ctxs_locs_by_terms):
        for ci, c in enumerate(t):
            for di, d in enumerate(c):
                if len(d):
                    while len(ctxs_locs_by_docs[di]) <= ci:
                        ctxs_locs_by_docs[di].append(
                            [[] for _ in range(terms_count)]
                        )
                    ctxs_locs_by_docs[di][ci][ti] = d
    # Note: d = doc, c = context word, t = term
    return ctxs_locs_by_docs


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
        ['hello', 'world'],
        ['good'],
        []
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

