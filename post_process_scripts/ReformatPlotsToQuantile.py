import pandas as pd

from utils import plot_dif_std

# names = ['pmelif_plots', 'pmelif_plots_op', 'pmelif_plots_both']
# directory = '../datasets/'
# for file_name in os.listdir(directory):  # open directory with datasets
#     subname = file_name[:-4]
#     for name in names:
df = pd.read_csv('../results/madelon/pmelif_plots/comparison.csv')

plot_dif_std(df, '../results/madelon/pmelif_plots/pmelif_plots/test.png', 'recall_score', 10)
