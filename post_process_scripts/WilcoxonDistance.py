import os

import numpy as np
import pandas as pd

names = ['pmelif_plots_op', 'pmelif_plots_both', 'pmelif_plots']
result = []
directory = '../datasets/'
for file_name in os.listdir(directory):  # open directory with datasets
    subname = file_name[:-4]
    for name in names:
        df = pd.read_csv('../results/' + subname + '/comparison_pear/' + name + '/pmelif_dataset.csv')
        melif_score = []
        pmelif_score = []
        for index, row in df.iterrows():
            model = row['model']
            if model == 'melif':
                melif_score.append(row['score'])
            else:
                pmelif_score.append(row['score'])
        melif_score = np.array(melif_score)
        pmelif_score = np.array(pmelif_score)
        distance = pmelif_score - melif_score
        zipped_distance = zip(np.abs(distance), np.sign(distance))
        sorted_distance = sorted(zipped_distance, key=lambda t: t[0])
        sum_distance = 0
        sum_positive = 0
        sum_negative = 0
        for j in range(len(sorted_distance)):
            sign_value = sorted_distance[j][1]
            if sorted_distance[j][1] == 1.0:
                sum_positive += j + 1
            elif sorted_distance[j][1] == -1.0:
                sum_negative += j + 1
            else:
                sign_value = 1
            sum_distance += sign_value * (j + 1)  # for 0.0 diff sign value should be eps or +-1
        normed_distance = sum_distance / max(sum_negative, sum_positive)
        result.append([subname + name, normed_distance])
result_df = pd.DataFrame(result, columns=['name', 'wilcoxon_distance'])
result_df.to_csv('../results/wilcoxon_distance.csv')
