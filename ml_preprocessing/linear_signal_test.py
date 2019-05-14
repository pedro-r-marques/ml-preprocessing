from .linear_signal import LinearDatasetAccessor, LinearSignalGenerator, SignalFeatureGenerator

import unittest
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal


class LinearDatasetAccessorTest(unittest.TestCase):
    def test_continuos_blocks_merge_one(self):
        df = pd.DataFrame()
        accessor = LinearDatasetAccessor(df, 10, [1, 3, 4, 6, 9])
        expected = [(1, 2), (3, 5), (6, 7), (9, 10)]
        self.assertListEqual(accessor.get_contiguos_blocks(), expected)

    def test_continuos_blocks_single_element(self):
        df = pd.DataFrame()
        accessor = LinearDatasetAccessor(df, 10, [4])
        expected = [(4, 5)]
        self.assertListEqual(accessor.get_contiguos_blocks(), expected)

    def test_continuos_blocks_merge_multiple(self):
        df = pd.DataFrame()
        accessor = LinearDatasetAccessor(df, 10, [0, 1, 2, 3, 4, 5, 6, 8, 9])
        expected = [(0, 7), (8, 10)]
        self.assertListEqual(accessor.get_contiguos_blocks(), expected)


class MockFeatureGenerator(SignalFeatureGenerator):
    def generate(self, df: pd.DataFrame, predict=False):
        return df.values.reshape(-1), np.array([0])


class LinearSignalGeneratorTest(unittest.TestCase):
    def test_basic(self):
        signal = np.random.rand(100)
        df = pd.DataFrame({'signal': signal})
        accessor = LinearDatasetAccessor(df, 1, [0])
        feature_gen = MockFeatureGenerator()
        generator = LinearSignalGenerator(
            accessor, 10, feature_gen, 5, batch_size=10)
        self.assertEqual(len(generator), 2)
        b0 = generator[0]
        b1 = generator[1]
        self.assertEqual(b0[0].shape, (10, 10))
        self.assertEqual(b1[0].shape, (9, 10))

    def test_no_overlap(self):
        """Ensure that there is no overlap between the data that a generator is
        allowed to access and data outside of its range.
        """
        signal = np.arange(1000)
        df = pd.DataFrame({'signal': signal})
        # Define 2 blocks of 300 and 200 data points respectivly
        accessor = LinearDatasetAccessor(df, 10, [1, 2, 3, 7, 8])
        feature_gen = MockFeatureGenerator()
        generator = LinearSignalGenerator(
            accessor, 10, feature_gen, 5, batch_size=10)
        self.assertEqual(len(generator), (59 + 39 - 1) // 10 + 1)

        r1_count = 0
        r2_count = 0
        for index in range(len(generator)):
            b = generator[index]
            for example in b[0]:
                self.assertTrue(np.all(np.diff(example) == 1))
                r1 = np.all(np.where((example >= 100) &
                                     (example < 400), True, False))
                r2 = np.all(np.where((example >= 700) &
                                     (example < 900), True, False))
                self.assertTrue(r1 or r2, example)
                if r1:
                    r1_count += 1
                if r2:
                    r2_count += 1
        self.assertEqual(r1_count, 59)
        self.assertEqual(r2_count, 39)

    def test_non_divisible_sizes(self):
        """Repeat the non_overlap test with sizes such that the number of
        datapoints is not divisible by the number of blocks.
        """
        signal = np.arange(1000)
        df = pd.DataFrame({'signal': signal})
        # Define 2 blocks of 250 and 125 data points respectivly
        accessor = LinearDatasetAccessor(df, 16, [1, 2, 3, 4, 7, 8])
        feature_gen = MockFeatureGenerator()
        generator = LinearSignalGenerator(
            accessor, 10, feature_gen, 5, batch_size=8)
        b0_examples = 24 * 2
        b1_examples = 12 * 2 - 1
        self.assertEqual(
            len(generator), (b0_examples + b1_examples - 1) // 8 + 1)

        r1_count = 0
        r2_count = 0
        for index in range(len(generator)):
            b = generator[index]
            for example in b[0]:
                self.assertTrue(np.all(np.diff(example) == 1))
                b0_start = 62
                b0_end = b0_start + 250
                r1 = np.all(np.where((example >= b0_start) &
                                     (example < b0_end), True, False))
                b1_start = 7 * 62
                b1_end = b1_start + 125
                r2 = np.all(np.where((example >= b1_start) &
                                     (example < b1_end), True, False))
                self.assertTrue(r1 or r2, example)
                if r1:
                    r1_count += 1
                if r2:
                    r2_count += 1
        self.assertEqual(r1_count, b0_examples)
        self.assertEqual(r2_count, b1_examples)

    def test_list_inputs(self):
        class ListFeatureGenerator(object):
            def generate(self, df: pd.DataFrame, predict=False):
                return [np.array([0]), np.array([1])], np.array([0])

        signal = np.random.rand(100)
        df = pd.DataFrame({'signal': signal})
        accessor = LinearDatasetAccessor(df, 1, [0])
        feature_gen = ListFeatureGenerator()
        generator = LinearSignalGenerator(
            accessor, 10, feature_gen, batch_size=10)
        self.assertEqual(len(generator), 1)
        actual = generator[0][0]
        expected = [np.array([0] * 10)[:, np.newaxis],
                    np.array([1] * 10)[:, np.newaxis]]
        self.assertTrue(isinstance(actual, list))
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            assert_array_equal(a, e)


if __name__ == '__main__':
    unittest.main()
