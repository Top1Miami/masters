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
from utils import select_k_best_abs
from utils.ComparisonUtils import convert_to_str
from utils.ComparisonUtils import estimator_score
from utils.ComparisonUtils import generate_sample
from utils.ComparisonUtils import generate_sampling_data
from utils.ComparisonUtils import split_features


class FourthPipelineExperiment(Experiment):
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
                 ensemble, save_path, alpha_start=0.1, alpha_end=1.1, alpha_delta=0.1,
                 points=None, delta=0.2, known_train_features=10, sample_size=100):
        self.generated_samples_number = generated_samples_number
        self.known_features_filter = known_features_filter
        self.ensemble = ensemble
        self.save_path = save_path
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_delta = alpha_delta
        self.points = points
        self.delta = delta
        self.known_train_features = known_train_features
        self.sample_size = sample_size

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
        alphas = np.arange(self.alpha_start, self.alpha_end, self.alpha_delta)
        univariate_filters, filter_names = FourthPipelineExperiment.algorithms(len(known_features))
        for sample_number in range(self.generated_samples_number):
            sample_indices_object = generate_sample(class_zero, class_first, count_zero, count_first)
            train_known_features, test_known_features = split_features(known_features, self.known_train_features)

            train_scoring = FourthPipelineExperiment.exact_rec_wrapper(train_known_features)
            test_scoring = FourthPipelineExperiment.exact_rec_wrapper(test_known_features)
            test_precision_scoring = FourthPipelineExperiment.exact_prec_wrapper(test_known_features)
            x = features[sample_indices_object]
            y = labels[sample_indices_object]
            for i in range(len(univariate_filters)):
                univariate_filter = univariate_filters[i]
                filter_name = filter_names[i]
                univariate_filter.fit(x, y)
                filter_scores = np.nan_to_num(univariate_filter.feature_scores_)

                filter_selected_features = select_k_best_abs(len(known_features))(filter_scores)
                filter_not_train = [feature for feature in filter_selected_features if
                                    feature not in train_known_features]

                filter_recall_score = test_scoring(filter_not_train)
                train_filter_recall_score = train_scoring(filter_selected_features)
                filter_estimator_score = estimator_score(x, y, SVC(), filter_selected_features)
                filter_precision_score = test_precision_scoring(filter_not_train)
                print('filter {0}, filter score {1}, filter features {2}, '.format(filter_name, filter_estimator_score,
                                                                                   filter_selected_features))
                accumulated_info.append([filter_recall_score, train_filter_recall_score,
                                         filter_precision_score, filter_estimator_score,
                                         convert_to_str(filter_not_train), filter_name, ''])
            # run pmelif
            for alpha in alphas:
                scoring_function = LinearCombinationScoring([1, alpha],
                                                            [ClassifierScoring('f1_macro'),
                                                             RecallFeatureScoring(train_scoring)])
                print('PMeLiF with alpha {0}'.format(alpha))
                pmelif = PMeLiF(SVC(),
                                scoring_function,
                                select_k_best_abs(len(known_features)), self.ensemble,
                                points=self.points, delta=self.delta)

                pmelif.fit(x, y)

                pmelif_selected_features = pmelif.selected_features_
                pmelif_not_train = [feature for feature in pmelif_selected_features if
                                    feature not in train_known_features]
                pmelif_recall_score = test_scoring(pmelif_not_train)
                train_pmelif_recall_score = train_scoring(pmelif_selected_features)
                pmelif_estimator_score = estimator_score(x, y, SVC(), pmelif_not_train)
                pmelif_precision_score = test_precision_scoring(pmelif_not_train)
                pmelif_best_point = pmelif.best_point_

                accumulated_info.append([pmelif_recall_score, train_pmelif_recall_score,
                                         pmelif_precision_score, pmelif_estimator_score,
                                         convert_to_str(pmelif_not_train), 'pmelif' + str(round(alpha, 2)),
                                         convert_to_str(pmelif_best_point)])
            # run melif
            print('MeLiF')
            melif = PMeLiF(SVC(), ClassifierScoring('f1_macro'), select_k_best_abs(len(known_features)),
                           self.ensemble,
                           points=self.points, delta=self.delta)

            melif.fit(x, y)

            melif_selected_features = melif.selected_features_
            melif_not_train = [feature for feature in melif_selected_features if
                               feature not in train_known_features]

            melif_recall_score = test_scoring(melif_not_train)
            train_melif_recall_score = train_scoring(melif_selected_features)
            melif_estimator_score = melif.best_score_
            melif_precision_score = test_precision_scoring(melif_not_train)
            melif_best_point = melif.best_point_
            print('melif best point {0}, melif score {1}, melif features {2}'.format(melif_best_point,
                                                                                     melif_estimator_score,
                                                                                     melif_selected_features))

            accumulated_info.append([melif_recall_score, train_melif_recall_score,
                                     melif_precision_score, melif_estimator_score,
                                     convert_to_str(melif_not_train), 'melif', convert_to_str(melif_best_point)])
            print('------------------------------Sample {0} end'.format(sample_number))

        df = pd.DataFrame(data=accumulated_info,
                          columns=['recall_score', 'train_recall_score', 'precision_score', 'estimator_score',
                                   'selected_features', 'model', 'points'])

        if not os.path.exists(self.save_path.format(subname)):
            os.makedirs(self.save_path.format(subname))

        model_names = ['melif'] + ['pmelif' + str(round(alpha, 2)) for alpha in alphas] + filter_names
        stats = []
        for model_name in model_names:
            melif_df = df.loc[df['model'] == model_name]
            recall_mean = round(melif_df['recall_score'].mean(), 2)
            recall_std = round(melif_df['recall_score'].std(), 2)

            precision_mean = round(melif_df['precision_score'].mean(), 2)
            precision_std = round(melif_df['precision_score'].std(), 2)

            estimator_mean = round(melif_df['estimator_score'].mean(), 2)
            estimator_std = round(melif_df['estimator_score'].std(), 2)

            train_recall_mean = round(melif_df['train_recall_score'].mean(), 2)
            train_recall_std = round(melif_df['train_recall_score'].std(), 2)
            stats.append(
                [recall_mean, recall_std, precision_mean, precision_std, estimator_mean, estimator_std,
                 train_recall_mean, train_recall_std, model_name])

        stats_df = pd.DataFrame(stats,
                                columns=['recall_mean', 'recall_std', 'precision_mean', 'precision_std',
                                         'estimator_mean', 'estimator_std',
                                         'train_recall_mean', 'train_recall_std', 'model_name'])
        stats_df.to_csv(self.save_path.format(subname) + 'statistics.csv')
        df.to_csv(self.save_path.format(subname) + 'score_dumps.csv')

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
        # filter_names = ['su', 'fechner', 'spearman', 'igain']
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
