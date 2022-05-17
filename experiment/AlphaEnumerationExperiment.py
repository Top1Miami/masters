import os.path

import numpy as np
import pandas as pd
from ITMO_FS import UnivariateFilter
from ITMO_FS import fechner_corr
from ITMO_FS import gini_index
from ITMO_FS import information_gain
from ITMO_FS import pearson_corr
from ITMO_FS import reliefF_measure
from ITMO_FS import spearman_corr
from ITMO_FS import su_measure
from sklearn.svm import SVC

from experiment import Experiment
from models import LinearCombinationScoring
from models import PMeLiF
from models.ScoringFunctions import ClassifierScoring
from models.ScoringFunctions import RecallFeatureScoring
from utils import anova_measure_scaled
from utils import chi2_measure_scaled
from utils import plot_single
from utils import select_k_best_abs
from utils.ComparisonUtils import convert_to_str
from utils.ComparisonUtils import estimator_score
from utils.ComparisonUtils import generate_sample
from utils.ComparisonUtils import generate_sampling_data
from utils.ComparisonUtils import split_features


class AlphaEnumerationExperiment(Experiment):
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
    sample_size : int
        number of objects to use for experiment sample
    """

    def __init__(self, generated_samples_number, known_features_filter,
                 ensemble, save_path, points=None, delta=0.2,
                 known_train_features=10, sample_size=100):
        self.generated_samples_number = generated_samples_number
        self.known_features_filter = known_features_filter
        self.ensemble = ensemble
        self.points = points
        self.delta = delta
        self.known_train_features = known_train_features
        self.sample_size = sample_size
        self.save_path = save_path

    def run(self, features, labels, file_name):
        print('Number of samples generated {0}, size of sample {1}'.format(self.generated_samples_number,
                                                                           self.sample_size))
        subname = file_name[:-4]
        df = pd.read_csv('../results/' + subname + '/selected_features/sel_features.csv')
        # df = pd.read_csv('/nfs/home/dshusharin/masters/results/' + subname + '/selected_features/sel_features.csv')
        known_features = [int(i) for i in
                          df.loc[df['filter_name'] == self.known_features_filter]
                          ['joined_features'].iloc[0].split(',')]

        class_zero, class_first, count_zero, count_first = generate_sampling_data(labels,
                                                                                  self.sample_size)

        accumulated_info = []
        for _ in range(self.generated_samples_number):
            sample_indices_object = generate_sample(class_zero, class_first, count_zero, count_first)
            train_known_features, test_known_features = split_features(known_features, self.known_train_features)

            train_scoring = AlphaEnumerationExperiment.exact_rec_wrapper(train_known_features)
            test_scoring = AlphaEnumerationExperiment.exact_rec_wrapper(test_known_features)
            test_precision_scoring = AlphaEnumerationExperiment.exact_prec_wrapper(test_known_features)
            x = features[sample_indices_object]
            y = labels[sample_indices_object]

            # run pmelif
            for alpha in np.arange(0.0, 1.0, 0.05):
                scoring_function = LinearCombinationScoring([1, alpha],
                                                            [ClassifierScoring('f1_macro'),
                                                             RecallFeatureScoring(train_scoring)])
                pmelif = PMeLiF(SVC(),
                                scoring_function,
                                select_k_best_abs(len(train_known_features)), self.ensemble,
                                points=self.points, delta=self.delta)

                pmelif.fit(x, y)
                pmelif_score_matrix = pmelif.pmelif_transform(x, y)

                pmelif_selected_features = select_k_best_abs(len(known_features))(pmelif_score_matrix)
                pmelif_not_train = [feature for feature in pmelif_selected_features if
                                    feature not in train_known_features]
                pmelif_recall_score = test_scoring(pmelif_not_train)
                pmelif_estimator_score = estimator_score(x, y, SVC(), pmelif_not_train)
                pmelif_precision_score = test_precision_scoring(pmelif_not_train)
                pmelif_best_point = pmelif.best_point_

                accumulated_info.append([pmelif_recall_score, pmelif_precision_score, pmelif_estimator_score,
                                         alpha, convert_to_str(pmelif_not_train), convert_to_str(pmelif_best_point)])

        df = pd.DataFrame(data=accumulated_info,
                          columns=['recall_score', 'precision_score', 'estimator_score',
                                   'alpha', 'selected_features', 'points'])

        if not os.path.exists(self.save_path.format(subname)):
            os.makedirs(self.save_path.format(subname))

        plot_single(df, self.save_path.format(subname) + 'alpha_feature_recall_std.png', 'recall_score')
        plot_single(df, self.save_path.format(subname) + 'alpha_estimator_score_std.png', 'estimator_score')
        plot_single(df, self.save_path.format(subname) + 'alpha_feature_precision_std.png', 'precision_score')

        df.to_csv(self.save_path.format(subname) + 'alpha_comparison.csv')

    @staticmethod
    def algorithms(number_of_features):
        univariate_filters = [
            UnivariateFilter(su_measure, select_k_best_abs(number_of_features)),
            UnivariateFilter(fechner_corr, select_k_best_abs(number_of_features)),
            UnivariateFilter(spearman_corr, select_k_best_abs(number_of_features)),
            UnivariateFilter(pearson_corr, select_k_best_abs(number_of_features)),
            UnivariateFilter(information_gain, select_k_best_abs(number_of_features)),
            UnivariateFilter(gini_index, select_k_best_abs(number_of_features)),
            UnivariateFilter(chi2_measure_scaled, select_k_best_abs(number_of_features)),
            UnivariateFilter(reliefF_measure, select_k_best_abs(number_of_features)),
            UnivariateFilter(anova_measure_scaled, select_k_best_abs(number_of_features))
        ]
        filter_names = ['su', 'fechner', 'spearman', 'pearson', 'igain', 'gini', 'chi2', 'reliefF', 'anova']
        return univariate_filters, filter_names

    @staticmethod
    def exact_prec_wrapper(known_features):
        def feature_prec(predicted_features):
            correct_guess = 0.
            for feature in predicted_features:
                if feature in known_features:
                    correct_guess += 1.
            return correct_guess / len(predicted_features)

        return feature_prec

    @staticmethod
    def exact_rec_wrapper(known_features):
        def feature_rec(predicted_features):
            correct_guess = 0.
            for feature in predicted_features:
                if feature in known_features:
                    correct_guess += 1
            return correct_guess / len(known_features)

        return feature_rec
