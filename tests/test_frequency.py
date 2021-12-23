import unittest


class FrequencyTestCase(unittest.TestCase):
    def test_something(self):
        # Input: locations, documents

        # For each term in the vocabulary:

        # 0. Normalize
        # Convert join all texts and treat sentences as documents.
        # Requires averaging at the end to achieve a single value per term.

        # 1. Calculate the term frequency for each document
        # where a document is a sentence in the corpus.

        # 2. Calculate the inverse document frequency for each term
        # by dividing the number of documents (sents) in the group by level
        # by the number of documents where the term appears.

        # 3. Calculate the TF-IDF for each document.

        # 4. Average the TF-IDF for each document and connect it with the term.

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
