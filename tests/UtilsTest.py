import unittest

import numpy as np

from utils import select_k_best_abs


class MyTestCase(unittest.TestCase):
    def test_best_abs(self):
        x = np.array([1, 7, 10, 3, 4, -10, 1, -3, -5, -9])
        self.assertEquals(set(x[select_k_best_abs(4)(x)]), {10, -10, -9, 7})
