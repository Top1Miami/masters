import pandas as pd

df = pd.read_csv('../results/madelon/comparison_pear/pmelif_plots_op/comparison.csv')

score_field = 'recall_score'

melif_df = df.loc[df['model'] == 'melif']
mq1_plot = []
mq3_plot = []
m_mean_plot = []

pmelif_df = df.loc[df['model'] == 'pmelif']
pmq1_plot = []
pmq3_plot = []
pm_mean_plot = []

for i in range(1, 11):
    print('Number of features to select {0}'.format(i))
    melif_scores = melif_df.loc[melif_df['features_number'] == i][score_field]
    print('MeLiF scores:')
    print(melif_scores.describe())
    print(melif_scores.value_counts())
    # mq1_plot.append(melif_scores.quantile(0.25))
    # mq3_plot.append(melif_scores.quantile(0.75))
    # m_mean_plot.append(melif_scores.mean())

    pmelif_scores = pmelif_df.loc[pmelif_df['features_number'] == i][score_field]
    print('PMeLiF scores:')
    print(pmelif_scores.describe())
    print(pmelif_scores.value_counts())
    # pmq1_plot.append(pmelif_scores.quantile(0.25))
    # pmq3_plot.append(pmelif_scores.quantile(0.75))
    # pm_mean_plot.append(pmelif_scores.mean())
