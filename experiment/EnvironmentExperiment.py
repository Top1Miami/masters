import os

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
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from experiment.Experiment import Experiment
from utils import anova_measure_scaled
from utils import chi2_measure_scaled
from utils import select_k_best_abs


class EnvironmentExperiment(Experiment):
    # su_measure fechner_corr spearman_corr pearson_corr information_gain
    # gini_index chi2_measure relief_measure reliefF_measure laplacian_score anova
    number_of_features = 20

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

    filter_names = [
        'su', 'fechner', 'spearman', 'pearson', 'igain', 'gini', 'chi2', 'reliefF', 'anova'
    ]

    def run(self, features, labels, file_name):
        result = []
        for i, univariate_filter in enumerate(EnvironmentExperiment.univariate_filters):
            univariate_filter.fit_transform(features, labels)
            classifier = SGDClassifier(max_iter=1000, tol=1e-3)
            print(univariate_filter)
            print("Selected features", sorted(univariate_filter.selected_features_))
            scores = cross_val_score(classifier, features[:, univariate_filter.selected_features_], labels)
            joined_features = ','.join([str(i) for i in univariate_filter.selected_features_])
            joined_scores = ','.join([str(round(univariate_filter.feature_scores_[i], 3)) for i in univariate_filter.selected_features_])
            result.append([EnvironmentExperiment.filter_names[i], np.mean(scores), np.std(scores), joined_features, joined_scores])

        df = pd.DataFrame(result, columns=['filter_name', 'mean_score', 'std_score', 'joined_features', 'joined_scores'])

        file_number = file_name[:-4]
        if not os.path.exists('../results/' + file_number + '/selected_features/'):
            os.makedirs('../results/' + file_number + '/selected_features/')
        df.to_csv('../results/' + file_number + '/selected_features/sel_features.csv')

    @staticmethod
    def difference(l1, l2):
        s1 = set(l1)
        s2 = set(l2)
        print("Number of matches ", len(s1.intersection(l2)))
        print("Number of old not selected", len(s2.difference(s1)))
        print("Not selected from old", s2.difference(s1))
