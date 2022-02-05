import unittest

import graded_readers_stats.data


class DataTestCase(unittest.TestCase):
    # prefix with 'x_' to skip the test
    def x_test_load(self):
        data = graded_readers_stats.data
        data.load(True, True, f'.{data.FOLDER_DATA_TRIAL}')
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
