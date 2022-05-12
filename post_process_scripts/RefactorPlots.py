import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

names = ['pmelif_plots', 'pmelif_plots_op', 'pmelif_plots_both']
directory = '../datasets/'
for file_name in os.listdir(directory):  # open directory with datasets
    subname = file_name[:-4]
    for name in names:
        df = pd.read_csv('../results/' + subname + '/comparison_pear/' + name + '/pmelif_dataset.csv')
        ax = sns.lineplot(x='features_number', y='score', hue='model',
                          data=df, ci='sd', palette=['red', 'blue'])
        ax.set_xticks(range(1, 11))
        ax.set_ylim([0.0, None])
        ax.set_xlim([1.0, None])
        plt.grid()
        plt.savefig('../results/' + subname + '/comparison_pear/' + name + '/pmelif_plot.png')
        plt.close()
