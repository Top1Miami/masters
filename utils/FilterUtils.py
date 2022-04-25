import numpy as np
from ITMO_FS import anova
from ITMO_FS import chi2_measure
from ITMO_FS.filters.univariate.measures import _wrapped_partial
from sklearn.preprocessing import MinMaxScaler


def select_k_best_abs(k):
    return _wrapped_partial(__select_k_abs, k=k, reverse=True)


def __select_k_abs(scores, k, reverse=False):
    if not isinstance(k, int):
        raise TypeError("Number of features should be integer")
    if k > scores.shape[0]:
        raise ValueError(
            "Cannot select %d features with n_features = %d" % (k, len(scores)))
    order = np.argsort(np.abs(scores))
    if reverse:
        order = order[::-1]
    return order[:k]


def chi2_measure_scaled(x, y):
    scores = chi2_measure(x, y)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(scores.reshape(len(scores), 1))
    return scaled_features.squeeze()


def anova_measure_scaled(x, y):
    scores = anova(x, y)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(scores.reshape(len(scores), 1))
    return scaled_features.squeeze()
