import unittest

import numpy as np

from experiment import ParMeLiFExperiment


class MyTestCase(unittest.TestCase):
    def test_subsampling(self):
        x = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1])
        class_zero, class_first, count_zero, count_first = ParMeLiFExperiment.generate_sampling_data(x, 4)
        print(class_zero)
        print(class_first)
        print(count_zero)
        print(count_first)
        print(ParMeLiFExperiment.generate_sample(class_zero, class_first, count_zero, count_first))
