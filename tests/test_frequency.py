from graded_readers_stats.frequency import freqs_by_term


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
