from graded_readers_stats.context import freqs_pipeline
from graded_readers_stats.frequency import freqs_by_term


def test_freqs_by_term_simple():
    # Given
    terms_locs = [
        [[(0, (0, 3))], []],
        [[], []],
    ]
    words_total = 10

    # When
    freqs = freqs_by_term(terms_locs, words_total)

    # Then
    assert [0.1, 0] == freqs


def test_freqs_by_term_complex():
    # Given
    s = (0, (0, 3))  # s is short for sentence
    terms_locs = [
        [[], [], [], []],
        [[s], [], [], []],
        [[s], [s], [], []],
        [[s], [s], [s], []],
        [[s, s], [s], [s], []],
        [[s, s, s], [s], [], []],
        [[s, s, s, s], [], [], []],
        [[s, s], [], [], [s, s]],
    ]
    words_total = 10

    # When
    freqs = freqs_by_term(terms_locs, words_total)

    # Then
    expected = [
        0 / words_total,
        1 / words_total,
        2 / words_total,
        3 / words_total,
        4 / words_total,
        4 / words_total,
        4 / words_total,
        4 / words_total,
        ]
    assert expected == freqs


def test_freqs_pipeline():
    # Given
    s = (0, (0, 2))  # some sentence occurrence
    terms_locs = [  # terms
        [  # term 1: avg freq = (0.4 + 0.1) / 2 = 0.5 / 2 = 0.25
            [  # ctx word 1: freq = 4 / 10 = 0.4
                [s, s],  # doc 1
                [s],     # doc 2
                [s],     # doc 3
            ],
            [  # ctx word 2: freq = 1 / 10 = 0.1
                [s],  # doc 1
                [],   # doc 2
                [],   # doc 3
            ]
        ],
        [  # term 2
            [  # ctx word 3: freq = 1 / 10 = 0.1
                [],   # doc 1
                [s],  # doc 2
                [],   # doc 3
            ],
        ],
    ]
    words_total = 10

    # When
    freqs = list(freqs_pipeline(words_total)(terms_locs))

    # Then
    assert [0.25, 0.1] == freqs
