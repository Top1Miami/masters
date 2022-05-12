import os

import matplotlib.pyplot as plt
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

from experiment import Experiment
from utils import anova_measure_scaled
from utils import chi2_measure_scaled
from utils import select_k_best_abs


class FeatureComparisonExperiment(Experiment):
    number_of_features = 20
    # su_measure fechner_corr spearman_corr pearson_corr information_gain
    # gini_index chi2_measure relief_measure reliefF_measure anova
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

    def __init__(self, generate_plots=False):
        self.generate_plots = generate_plots

    class CrossRating:
        class SingleLine:
            def __init__(self, left_score, right_score, color):
                self.left_score = left_score
                self.right_score = right_score
                self.color = color

            def get_line(self):
                return [self.left_score, self.right_score], self.color

        def __init__(self):
            self.lines = []

        def add_line(self, left_score, right_score, color):
            self.lines.append(self.SingleLine(left_score, right_score, color))

        def get_lines(self):
            return self.lines

    def run(self, features, labels, file_name):

        [univariate_filter.fit(features, labels) for univariate_filter in
         FeatureComparisonExperiment.univariate_filters]
        for i, univariate_filter in enumerate(FeatureComparisonExperiment.univariate_filters):
            univariate_filter.fit(features, labels)
            print('Done {0}'.format(FeatureComparisonExperiment.filter_names[i]))

        # [print(univariate_filter.feature_scores_) for univariate_filter in univariate_filters]

        distances = []
        for i in range(0, len(FeatureComparisonExperiment.univariate_filters)):
            for j in range(i + 1, len(FeatureComparisonExperiment.univariate_filters)):
                left_filter = FeatureComparisonExperiment.univariate_filters[i]
                right_filter = FeatureComparisonExperiment.univariate_filters[j]

                left_selected_features = left_filter.selected_features_
                right_selected_features = right_filter.selected_features_

                left_scores = left_filter.feature_scores_
                right_scores = right_filter.feature_scores_

                cross_rating = self.CrossRating()

                features_selected_both = 0
                features_selected_left = 0
                features_selected_right = 0
                only_left_features = []
                only_right_features = []
                for selected_feature in left_selected_features:
                    left_score = abs(left_scores[selected_feature])
                    right_score = abs(right_scores[selected_feature])
                    if selected_feature in right_selected_features:
                        color = 'red'
                        features_selected_both += 1
                    else:
                        color = 'yellow'
                        features_selected_left += 1
                        only_left_features.append(selected_feature)
                    cross_rating.add_line(left_score, right_score, color)

                for selected_feature in right_selected_features:
                    if selected_feature not in left_selected_features:
                        left_score = abs(left_scores[selected_feature])
                        right_score = abs(right_scores[selected_feature])
                        cross_rating.add_line(left_score, right_score, 'green')
                        features_selected_right += 1
                        only_right_features.append(selected_feature)

                if self.generate_plots:
                    FeatureComparisonExperiment.gen_plots(features_selected_both, features_selected_left,
                                                          features_selected_right,
                                                          i, j, cross_rating, file_name)
                    print(
                        'Generated plots for filters {0} against {1}'.format(
                            FeatureComparisonExperiment.filter_names[i],
                            FeatureComparisonExperiment.filter_names[j]
                        ))
                print('Generating distances for filter {0} against {1}'.format(
                    FeatureComparisonExperiment.filter_names[i],
                    FeatureComparisonExperiment.filter_names[j]
                ))
                distance_info = FeatureComparisonExperiment.prepare_distance_info(only_left_features,
                                                                                  only_right_features,
                                                                                  left_filter, right_filter)

                distances.append([FeatureComparisonExperiment.filter_names[i],
                                  FeatureComparisonExperiment.filter_names[j],
                                  len(only_left_features)] + distance_info)

        distance_dump = FeatureComparisonExperiment.count_distances(distances)

        file_number = file_name[:-4]
        if not os.path.exists('../results/' + file_number + '/distance/'):
            os.makedirs('../results/' + file_number + '/distance/')
        distance_dump.to_csv('../results/' + file_number + '/distance/distance_dump.csv')

    @staticmethod
    def count_distances(distances):
        dump_columns = ['left_name', 'right_name', 'feature_difference', 'left_max_value', 'left_max_ind',
                        'left_min_value', 'left_min_ind',
                        'right_on_left_max_value', 'right_on_left_max_ind', 'right_on_left_min_value',
                        'right_on_left_min_ind', 'right_max_value', 'right_max_ind', 'right_min_value',
                        'right_min_ind', 'left_on_right_max_value',
                        'left_on_right_max_ind', 'left_on_right_min_value', 'left_on_right_min_ind']

        distance_dump = pd.DataFrame(distances, columns=dump_columns)

        # outer distance between lower boundary of selected features and upper boundary of other filter selected features
        distance_dump['diff_left_outer'] = distance_dump['right_on_left_max_ind'] - distance_dump['left_min_ind']
        distance_dump['diff_right_outer'] = distance_dump['left_on_right_max_ind'] - distance_dump['right_min_ind']
        del_left_outer = max(distance_dump['diff_left_outer'])
        del_right_outer = max(distance_dump['diff_right_outer'])

        # inner distance between upper and lower boundary of other filter selected features
        distance_dump['diff_left_inner'] = distance_dump['right_on_left_min_ind'] - distance_dump[
            'right_on_left_max_ind']
        distance_dump['diff_right_inner'] = distance_dump['left_on_right_min_ind'] - distance_dump[
            'left_on_right_max_ind']
        del_left_inner = max(distance_dump['diff_left_inner'])
        del_right_inner = max(distance_dump['diff_right_inner'])

        # distance formula dist = log (normalized outer distance) + log(normalized inner distance) + feature number difference
        distance_dump['distance'] = np.log(1 + distance_dump['diff_left_outer'] / del_left_outer) + \
                                    np.log(1 + distance_dump['diff_right_outer'] / del_right_outer) + \
                                    np.log(1 + distance_dump['diff_left_inner'] / del_left_inner) + \
                                    np.log(1 + distance_dump['diff_right_inner'] / del_right_inner) + \
                                    distance_dump['feature_difference'] / \
                                    (FeatureComparisonExperiment.number_of_features - distance_dump[
                                        'feature_difference'])

        distance_dump.loc[distance_dump['feature_difference'] == FeatureComparisonExperiment.number_of_features,
                          'distance'] = (float('inf'))

        distance_dump = distance_dump.sort_values('distance', ascending=False)

        return distance_dump.round(3)

    @staticmethod
    def prepare_distance_info(selected_features_left, selected_features_right, left_filter, right_filter):
        if len(selected_features_left) == 0:  # in case all features are similar the distance should be 0
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if len(selected_features_left) == FeatureComparisonExperiment.number_of_features:  # in case all features are different distance metric will be inf
            return [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
                    float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'),
                    float('inf'), float('inf')]
        left_sorted_features = np.argsort(np.abs(left_filter.feature_scores_))[::-1]
        left_mapping = {}
        for i, feature in enumerate(left_sorted_features):
            left_mapping[feature] = i
        right_sorted_features = np.argsort(np.abs(right_filter.feature_scores_))[::-1]
        right_mapping = {}
        for i, feature in enumerate(right_sorted_features):
            right_mapping[feature] = i

        left_max = 0, 0  # left max for normalizing
        left_min = float('inf'), 0  # left min for distance counting

        left_on_right_max = 0, 0  # left on right max for distance counting
        left_on_right_min = float('inf'), 0  # left on right min for normalizing
        for selected_feature in selected_features_left:
            left_abs = abs(left_filter.feature_scores_[selected_feature])
            right_abs = abs(right_filter.feature_scores_[selected_feature])

            if left_max[0] < left_abs:
                left_max = left_abs, left_mapping[selected_feature]
            if left_min[0] > left_abs:
                left_min = left_abs, left_mapping[selected_feature]
            if left_on_right_max[0] < right_abs:
                left_on_right_max = right_abs, right_mapping[selected_feature]
            if left_on_right_min[0] > right_abs:
                left_on_right_min = right_abs, right_mapping[selected_feature]

        right_max = 0, 0  # right max for normalizing
        right_min = float('inf'), 0  # right min for distance counting

        right_on_left_max = 0, 0  # right on left max for distance counting
        right_on_left_min = float('inf'), 0  # right on left min for normalizing
        for selected_feature in selected_features_right:
            left_abs = abs(left_filter.feature_scores_[selected_feature])
            right_abs = abs(right_filter.feature_scores_[selected_feature])

            if right_max[0] < right_abs:
                right_max = right_abs, right_mapping[selected_feature]
            if right_min[0] > right_abs:
                right_min = right_abs, right_mapping[selected_feature]
            if right_on_left_max[0] < left_abs:
                right_on_left_max = left_abs, left_mapping[selected_feature]
            if right_on_left_min[0] > left_abs:
                right_on_left_min = left_abs, left_mapping[selected_feature]
        return [left_max[0], left_max[1], left_min[0], left_min[1], right_on_left_max[0], right_on_left_max[1],
                right_on_left_min[0], right_on_left_min[1], right_max[0], right_max[1], right_min[0], right_min[1],
                left_on_right_max[0], left_on_right_max[1], left_on_right_min[0], left_on_right_min[1]]

    @staticmethod
    def gen_plots(features_selected_both, features_selected_left, features_selected_right,
                  i, j, cross_rating, file_name):
        plt.grid(visible=True)
        legend = 'Red stands for features in both top {0}, Yellow stands in {1} top {0}, Green stands in {2} top {0}. ' \
            .format(FeatureComparisonExperiment.number_of_features, FeatureComparisonExperiment.filter_names[i],
                    FeatureComparisonExperiment.filter_names[j])

        color_map = {'red': 'Both {0}'.format(features_selected_both),
                     'yellow': 'Left {0}'.format(features_selected_left),
                     'green': 'Right {0}'.format(features_selected_right)}

        fig = plt.figure(figsize=(8, 5))
        plt.grid(visible=True)
        fig.suptitle(legend, fontsize=10, x=0.5, y=0.05)
        fig.subplots_adjust(top=0.95, bottom=0.15)

        color_set = set()
        for line in cross_rating.get_lines():
            values, color = line.get_line()
            if color not in color_set:
                plt.plot(values, marker='o', color=color, label=color_map[color])
                color_set.add(color)
            else:
                plt.plot(values, marker='o', color=color)
        plt.legend()

        subname = file_name[:-4]
        if not os.path.exists('../results/' + subname + '/feature_plots/'):
            os.makedirs('../results/' + subname + '/feature_plots/')
        plt.savefig(
            '../results/' + subname + '/feature_plots/' + FeatureComparisonExperiment.filter_names[i]
            + FeatureComparisonExperiment.filter_names[j] + '.png'
        )
        plt.close(fig)
        plt.clf()
