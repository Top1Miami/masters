import numpy as np

from ITMO_FS.utils.data_check import *
from sklearn.model_selection import ParameterGrid


class PMeLiF:
    __filters = []  # filters used for composition
    __feature_names = []  # in case features are named
    __filter_weights = []  # weights of the filters in composition
    __estimator = None  # classifier used for training and testing provided dataset
    __points = []  # list of filter weights walked through while reaching the best solution
    __cutting_rule = None  # cutting rule for features to be selected
    __delta = None  # the value for filter_weight to be increased or decreased by
    _train_x = _train_y = _test_x = _test_y = None  # the training and testing parts of the dataset

    def __init__(self, filters, score=None):
        # check_filters(filters)
        self.__filters = filters  # filters used for composition
        self.__score = score  # score metric for evaluating feature selection quality
        self.best_score = 0  # best accumulated score during the work of Melif
        self.best_point = np.array(
            [1 / len(filters) for _ in range(len(filters))])  # best weight vector for filters in composition
        self.best_f = None
        self.iter_numbers = np.array([], dtype=np.integer)

    def get_best_point(self):
        return self.best_point

    def fit(self, X, y, cutting_rule, delta=0.5, feature_names=None, points=None):
        self.__feature_names = generate_features(X, feature_names)  # initialize the feature names
        self.__filter_weights = np.ones(len(self.__filters)) / len(
            self.__filters)  # initialize the weights with starting value
        self.__points = points  # list of points to start with if given
        self.__cutting_rule = cutting_rule  # cutting rule for selection features

        self.__delta = delta  # delta if given by default 0.gina is used

        check_cutting_rule(cutting_rule)  # check if cutting rule is suitable
        self._train_x, self._train_y = X, y  # training and testing datasets

    def run(self):  # run the melif feature selecting algorithm
        nu = {i: [] for i in self.__feature_names}  # dictionary for list of feature scores
        for _filter in self.__filters:  # iterate through filters
            _filter.fit_transform(self._train_x, self._train_y,
                                  feature_names=self.__feature_names, store_scores=True)  # fit the given filters
            for key, value in _filter.feature_scores.items():  # iterate through feature scores
                _filter.feature_scores[key] = abs(value)  # take the absolute values as some filters such as pearson
                # have negative scores showing the dependency between features and labels
            _min = min(_filter.feature_scores.values())  # take the minimum score value
            _max = max(_filter.feature_scores.values())  # take the maximum score value
            for key, value in _filter.feature_scores.items():  # iterate through filter scores
                nu[key].append((value - _min) / (_max - _min))  # append to feature score list normalized scores
        if self.__points is None:  # if no points are given
            self.__points = [self.__filter_weights]  # start with initial weights
        if isinstance(self.__points, ParameterGrid):
            self.__points = map(lambda d: list(d.values()), list(self.__points))
        for point in self.__points:
            self.__search(point, nu)  # perform the search for best filter weights

        return self.best_f  # return selected features

    def __search(self, point, features):  # search for feature weights
        iter_number = 0
        points = [point]  # points to evaluate
        while len(points) > 0:  # walk through the points
            iter_number += 1
            cur_point = points.pop()  # take last added point
            values = list(features.values())  # feature score lists
            n = dict(zip(features.keys(),
                         self.__measure(np.array(values), cur_point)))  # calculate the composed score of the feature
            keys = self.__cutting_rule(n)  # select the features according to the composed scores
            new_features = {i: features[i] for i in keys}  # dictionary selected feature number -> its original name
            score = self.__score(keys)
            if score > self.best_score:  # if current score is better
                self.best_score = score  # save current score as best
                self.best_point = cur_point  # save current filter weight vector as best
                self.best_f = new_features  # save selected features as best
                points += self.__get_candidates(cur_point, self.__delta)  # get other candidates for best solution
        self.iter_numbers = np.append(self.iter_numbers, iter_number)

    @staticmethod
    def __get_candidates(point, delta=0.1):
        tiled_points = np.tile(point, (len(point) * 2, 1))
        stacked = np.vstack((np.eye(len(point)) * delta, np.eye(len(point)) * -delta))
        for i in range(tiled_points.shape[0]):
            for j in range(tiled_points.shape[1]):
                if tiled_points[i][j] + stacked[i][j] < 0.:
                    tiled_points[i][j] = 0.
                elif tiled_points[i][j] + stacked[i][j] > 1.0:
                    tiled_points[i][j] = 1.0
                else:
                    tiled_points[i][j] += stacked[i][j]
        return list(tiled_points)

    @staticmethod
    def __measure(nu, weights):
        return np.dot(nu, weights)

    def transform(self, X, y, cutting_rule,
                  feature_names=None):  # select features according to current filter weight vector
        features = generate_features(X, feature_names)  # generate feature names if none given

        nu = {i: [] for i in features}  # dictionary for list of feature scores
        for _filter in self.__filters:  # iterate through filters
            _filter.fit(X, y, feature_names, store_scores=True)  # fit the given filters
            for key, value in _filter.feature_scores.items():  # iterate through feature scores
                _filter.feature_scores[key] = abs(value)  # take the absolute values as some filters such as pearson
                # have negative scores showing the dependency between features and labels
            _min = min(_filter.feature_scores.values())  # take the minimum score value
            _max = max(_filter.feature_scores.values())  # take the maximum score value
            for key, value in _filter.feature_scores.items():  # iterate through filter scores
                nu[key].append((value - _min) / (_max - _min))  # append to feature score list normalized scores
        n = dict(zip(nu.keys(), self.__measure(np.array(list(nu.values())),
                                               self.best_point)))  # calculate the composed score of the feature
        keys = cutting_rule(n)  # select the features according to the composed scores
        new_features = {i: nu[i] for i in keys}  # dictionary selected feature number -> its original name
        return list(new_features.keys())
