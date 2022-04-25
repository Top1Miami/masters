import random
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class MyTestCase(unittest.TestCase):
    def test_read(self):
        class_name = 'class'
        df = pd.DataFrame([[5, 2], [6, 1], [7, 1], [8, 2], [9, 2]], columns=['f1', 'class'])
        print(df)
        labels = df[class_name]
        features = df.drop(class_name, axis=1)
        size = 3.
        ind1 = list(np.where(labels.to_numpy() == 1)[0])
        ind2 = list(np.where(labels.to_numpy() == 2)[0])
        size1 = round(float(len(ind1)) / labels.size * size)
        size2 = round(float(len(ind2)) / labels.size * size)
        print("size", size1, size2)
        print(ind1)
        print(ind2)
        print(random.sample(ind1, size1))
        print(random.sample(ind2, size2))


if __name__ == '__main__':
    unittest.main()
