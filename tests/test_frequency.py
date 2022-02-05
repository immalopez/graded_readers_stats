import math
import unittest
import pandas as pd
from graded_readers_stats.frequency import tfidfs_for_groups, tfidfs


class FrequencyTestCase(unittest.TestCase):
    def test_tfidf3(self):
        # Given
        vocab_d = {
            'Lexical item': [
                'multi word 1',
                'word1'
            ],
            'Locations': [
                [[(0, (0, 3))], []],
                [[], []],
            ]}
        vocab_df = pd.DataFrame(vocab_d)

        docs_d = {
            'Topic': [
                'Biology',
                'Medicine',
            ],
            'Lemma': [
                [['multi', 'word', '1', 'word9', 'word10', '.'],
                 ['word11', 'word12', 'word15', 'word16', '.']],
                [['word13', 'word14', '.']],
            ]
        }
        docs_df = pd.DataFrame(docs_d)

        # When
        vocab_tfidfs = tfidfs(
            vocab_locs=vocab_df['Locations'],
            docs=docs_df
        )

        # Then
        self.assertEqual(0.018814374728998825, vocab_tfidfs[0])
        self.assertEqual(0.0, vocab_tfidfs[1])

    def test_tfidf2(self):
        # Given
        vocab_d = {
                    'Lexical item': [
                        'multi word 1',
                        'word1'
                    ],
                    'Locations': [
                        [[], []],
                        [[(0, (0, 1))], [(0, (0, 1))]],
                    ]}
        vocab_df = pd.DataFrame(vocab_d)

        docs_d = {
            'Topic': [
                'Biology',
                'Medicine',
            ],
            'Lemma': [
                [['word1', 'word2', '.'],
                 ['word3', 'word4', '.'],
                 ['word5', 'word6', '.'],
                 ['word7', 'word8', '.']],
                [['word1', 'word4', '.']],
            ]
        }
        docs_df = pd.DataFrame(docs_d)

        # When
        vocab_tfidfs = tfidfs(
            vocab_locs=vocab_df['Locations'],
            docs=docs_df
        )

        # Then
        self.assertEqual(0.0, vocab_tfidfs[0])
        self.assertEqual(0.025085832971998432, vocab_tfidfs[1])

    def test_tfidf(self):
        # Given
        vocab_d = {
                   'Lexical item': [
                       'multi word 1',
                       'word1'
                   ],
                   'Locations': [
                       [[], [(0, (0, 3))], [], []],
                       [[(0, (0, 1))], [], [], [(0, (0, 1))]],
                   ]}
        vocab_df = pd.DataFrame(vocab_d)

        docs_d = {
                  'Level': [
                      'Inicial',
                      'Avanzado',
                      'Inicial',
                      'Avanzado'
                  ],
                  'Lemma': [
                      [['word1', 'word2', '.'],
                       ['word3', 'word4', '.'],
                       ['word5', 'word6', '.'],
                       ['word7', 'word8', '.']],
                      [['multi', 'word', '1', 'word9', 'word10', '.'],
                       ['word11', 'word12', 'word15', 'word16', '.']],
                      [['word3', 'word4', '.']],
                      [['word13', 'word14', '.']],
                  ]}
        docs_df = pd.DataFrame(docs_d)

        # When
        result = tfidfs_for_groups(
            locs=vocab_df['Locations'],
            doc_groups=docs_df.groupby('Level'),
            column_id='Reader'
        )

        # Then
        prefix = 'TFIDF_Reader_'
        self.assertEqual(0.0, result[prefix + 'Inicial'][0])
        self.assertEqual(0.025085832971998432, result[prefix + 'Inicial'][1])
        self.assertEqual(0.018814374728998825, result[prefix + 'Avanzado'][0])
        self.assertEqual(0.0, result[prefix + 'Avanzado'][1])


if __name__ == '__main__':
    unittest.main()
