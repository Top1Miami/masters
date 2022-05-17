import abc
from enum import Enum

import numpy as np
from sklearn.model_selection import cross_val_score


class Mode(Enum):
    MELIF = 1
    PMELIF = 2
    COMBINED = 3


class ScoringFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def measure(self, X, y, selected_features, estimator, cv):
        pass

    @abc.abstractmethod
    def type(self):
        pass


class ClassifierScoring(ScoringFunction):
    def __init__(self, classifier_scoring):
        self.classifier_scoring = classifier_scoring

    def measure(self, X, y, selected_features, estimator, cv):
        return cross_val_score(
            estimator, X[:, selected_features], y, cv=cv,
            scoring=self.classifier_scoring).mean()

    def type(self):
        return Mode.MELIF


class RecallFeatureScoring(ScoringFunction):
    def __init__(self, recall):
        self.recall = recall

    def measure(self, X, y, selected_features, estimator, cv):
        return self.recall(selected_features)

    def type(self):
        return Mode.PMELIF


class LinearCombinationScoring(ScoringFunction):
    def __init__(self, coefficients, scoring_functions):
        self.coefficients = coefficients
        self.scoring_functions = scoring_functions

    def measure(self, X, y, selected_features, estimator, cv):
        return sum(
            np.array([scoring_function.measure(X, y, selected_features, estimator, cv) for scoring_function in
                      self.scoring_functions])
            *
            self.coefficients)

    def type(self):
        return Mode.COMBINED
