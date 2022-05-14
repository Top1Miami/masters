import os.path
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from experiment import Experiment
from models import PMeLiF
from models.ScoringFunctions import ClassifierScoring
from models.ScoringFunctions import RecallFeatureScoring
from utils import plot_std
from utils import select_k_best_abs


class PMeLiFExperiment(Experiment):
    """
    generated_samples_number : int,
        number of samples to run experiments on
    known_features_filter : str,
        baseline features filter, f.e. pearson, igain
    ensemble : list,
        list of UnivariateFilters
    max_features_select : int,
        number of features to select with MeLiF
    save_path : str,
        path to save result png which should take file number as an argument, f.e. folder/{0}/
    points : list,
        list of filter initial weight in MeLiF
    delta : float,
        delta used in modifying points in MeLiF grid search
    known_train_features : int,
        number of known train batch features
    subsample_size : int
        number of objects to use for experiment sample
    """

    def __init__(self, generated_samples_number, known_features_filter,
                 ensemble, max_features_select, save_path, points=None, delta=0.2,
                 known_train_features=10, subsample_size=100):
        self.generated_samples_number = generated_samples_number
        self.known_features_filter = known_features_filter
        self.ensemble = ensemble
        self.max_features_select = max_features_select
        self.points = points
        self.delta = delta
        self.known_train_features = known_train_features
        self.subsample_size = subsample_size
        if known_train_features < max_features_select:
            raise ValueError('Max features to select should be less than number of train known features.')
        self.save_path = save_path

    def run(self, features, labels, file_name):
        subname = file_name[:-4]
        df = pd.read_csv('../results/' + subname + '/selected_features/sel_features.csv')
        # df = pd.read_csv('/nfs/home/dshusharin/masters/results/' + subname + '/selected_features/sel_features.csv')
        known_features = [int(i) for i in
                          df.loc[df['filter_name'] == self.known_features_filter]
                          ['joined_features'].iloc[0].split(',')]

        class_zero, class_first, count_zero, count_first = PMeLiFExperiment.generate_sampling_data(labels,
                                                                                                   self.subsample_size)
        other_features = [i for i in range(0, len(features[0])) if i not in known_features]

        accumulated_info = []
        for _ in range(self.generated_samples_number):
            sample_indices_object = PMeLiFExperiment.generate_sample(class_zero, class_first, count_zero,
                                                                     count_first)
            train_known_features, test_known_features = PMeLiFExperiment.split_features(known_features,
                                                                                        self.known_train_features)
            train_other_features, test_other_features = PMeLiFExperiment.split_features(other_features,
                                                                                        int(len(
                                                                                            other_features) / 2))

            train_joined_features = PMeLiFExperiment.join_features(train_other_features, train_known_features)
            test_joined_features = PMeLiFExperiment.join_features(test_other_features, test_known_features)
            print('Number of joined train features {0}, test {1}'.format(len(train_joined_features),
                                                                         len(test_joined_features)))

            train_mapping = PMeLiFExperiment.generate_mapping(train_joined_features, train_known_features)
            test_mapping = PMeLiFExperiment.generate_mapping(test_joined_features, test_known_features)

            train_scoring = PMeLiFExperiment.feature_rec_wrapper(train_mapping, train_known_features)
            test_scoring = PMeLiFExperiment.feature_rec_wrapper(test_mapping, test_known_features)
            test_precision_scoring = PMeLiFExperiment.feature_prec_wrapper(test_mapping)

            train_x = features[sample_indices_object][:, train_joined_features]
            test_x = features[sample_indices_object][:, test_joined_features]
            y = labels[sample_indices_object]
            print('Shape of train x {0}, test x {1}, y {2}'.format(train_x.shape, test_x.shape, y.shape))
            for features_select in range(1, self.max_features_select + 1):
                # run pmelif
                pmelif = PMeLiF(LinearRegression(),
                                RecallFeatureScoring(train_scoring),
                                select_k_best_abs(features_select), self.ensemble,
                                points=self.points, delta=self.delta)

                pmelif.fit(train_x, y)
                pmelif_score_matrix = pmelif.pmelif_transform(test_x, y)

                pmelif_selected_features = select_k_best_abs(features_select)(pmelif_score_matrix)
                pmelif_recall_score = test_scoring(pmelif_selected_features)
                pmelif_estimator_score = PMeLiFExperiment.estimator_score(test_x, y, SVC(), pmelif_selected_features)
                pmelif_precision_score = test_precision_scoring(pmelif_selected_features)

                accumulated_info.append([pmelif_recall_score, pmelif_precision_score, pmelif_estimator_score,
                                         features_select, PMeLiFExperiment.convert_to_str(pmelif_selected_features),
                                         'pmelif'])
                # run melif
                melif = PMeLiF(SVC(), ClassifierScoring('f1_macro'), select_k_best_abs(features_select), self.ensemble,
                               points=self.points, delta=self.delta)

                melif.fit(train_x, y)
                melif_score_matrix = melif.pmelif_transform(test_x, y)

                melif_selected_features = select_k_best_abs(features_select)(melif_score_matrix)
                melif_recall_score = test_scoring(melif_selected_features)
                melif_estimator_score = PMeLiFExperiment.estimator_score(test_x, y, SVC(), melif_selected_features)
                melif_precision_score = test_precision_scoring(melif_selected_features)

                accumulated_info.append([melif_recall_score, melif_precision_score, melif_estimator_score,
                                         features_select, PMeLiFExperiment.convert_to_str(melif_selected_features),
                                         'melif'])
        df = pd.DataFrame(data=accumulated_info,
                          columns=['recall_score', 'precision_score', 'estimator_score', 'features_number',
                                   'selected_features', 'model'])

        if not os.path.exists(self.save_path.format(subname)):
            os.makedirs(self.save_path.format(subname))

        plot_std(df, self.save_path.format(subname) + 'feature_recall_std.png', 'recall_score',
                 self.max_features_select)
        plot_std(df, self.save_path.format(subname) + 'estimator_score_std.png', 'estimator_score',
                 self.max_features_select)
        plot_std(df, self.save_path.format(subname) + 'feature_precision_std.png', 'precision_score',
                 self.max_features_select)
        df.to_csv(self.save_path.format(subname) + 'comparison.csv')

    @staticmethod
    def convert_to_str(x):
        return ','.join([str(i) for i in x])

    @staticmethod
    def estimator_score(X, y, estimator, selected_features):
        return ClassifierScoring('f1_macro').measure(X, y, selected_features, estimator, 3)

    @staticmethod
    def feature_prec_wrapper(mapping):
        def feature_prec(predicted_features):
            correct_guess = 0.
            for feature in predicted_features:
                if feature in mapping:
                    correct_guess += 1.
            return correct_guess / len(predicted_features)

        return feature_prec

    @staticmethod
    def feature_rec_wrapper(mapping, known_features):
        def feature_rec(predicted_features):
            correct_guess = 0.
            for feature in predicted_features:
                if feature in mapping:
                    correct_guess += 1
            return correct_guess / len(known_features)

        return feature_rec

    @staticmethod
    def generate_mapping(features, known_features):
        mapping = {}
        for i in range(0, len(features)):
            if features[i] in known_features:
                mapping[i] = features[i]
        return mapping

    @staticmethod
    def join_features(l1, l2):
        l3 = l1 + l2
        random.shuffle(l3)
        return l3

    @staticmethod
    def split_features(features, train=10):
        train_features = random.sample(features, train)
        test_features = [i for i in features if i not in train_features]
        print(
            'Number of train features requested {0}, number of train features generated {1}, number of test features generate {2}'.format(
                train, len(train_features), len(test_features))
        )
        return train_features, test_features

    @staticmethod
    def generate_sampling_data(labels, subsample_size):
        class_zero = np.where(labels == 0)[0]
        class_first = np.where(labels == 1)[0]
        count_zero = int(float(len(class_zero)) / len(labels) * subsample_size)
        count_first = int(float(len(class_first)) / len(labels) * subsample_size)
        print('Number of objects', len(labels), 'Number of zero class objects', len(class_zero),
              'Number of first class objects', len(class_first))
        print('Number of scaled zero class', count_zero, 'Number of scaled first class', count_first)
        return class_zero, class_first, count_zero, count_first

    @staticmethod
    def generate_sample(class_zero, class_first, count_zero, count_first):
        zero = np.random.choice(class_zero, count_zero)
        first = np.random.choice(class_first, count_first)
        return [*zero, *first]
