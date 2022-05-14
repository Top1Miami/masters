import os

import pandas as pd
from matplotlib import pyplot as plt

names = ['pmelif_plots', 'pmelif_plots_op', 'pmelif_plots_both']
directory = '../datasets/'
for file_name in os.listdir(directory):  # open directory with datasets
    subname = file_name[:-4]
    for name in names:
        df = pd.read_csv('../results/' + subname + '/comparison_pear/' + name + '/pmelif_dataset.csv')

        melif_df = df.loc[df['model'] == 'melif']
        mq1_plot = []
        mq3_plot = []
        m_mean_plot = []

        pmelif_df = df.loc[df['model'] == 'pmelif']
        pmq1_plot = []
        pmq3_plot = []
        pm_mean_plot = []

        for i in range(1, 11):
            melif_scores = melif_df.loc[melif_df['features_number'] == i]['score']
            mq1_plot.append(melif_scores.quantile(0.25))
            mq3_plot.append(melif_scores.quantile(0.75))
            m_mean_plot.append(melif_scores.mean())

            pmelif_scores = pmelif_df.loc[pmelif_df['features_number'] == i]['score']
            pmq1_plot.append(pmelif_scores.quantile(0.25))
            pmq3_plot.append(pmelif_scores.quantile(0.75))
            pm_mean_plot.append(pmelif_scores.mean())

        ax = plt.gca()

        plt.plot(range(1, 11), mq1_plot, color='red')
        plt.plot(range(1, 11), mq3_plot, color='red')
        plt.plot(range(1, 11), m_mean_plot, label='melif', color='red')
        plt.plot(range(1, 11), pmq1_plot, color='blue')
        plt.plot(range(1, 11), pmq3_plot, color='blue')
        plt.plot(range(1, 11), pm_mean_plot, label='pmelif', color='blue')

        ax.legend()
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0.0, None])
        ax.set_xlim([1.0, None])
        plt.grid()
        plt.savefig('../results/' + subname + '/comparison_pear/' + name + '/pmelif_quantile.png')
        plt.close()
