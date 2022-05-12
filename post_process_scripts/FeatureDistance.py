import os

import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv('../results/bioresponse/distance/distance_dump.csv')
# pr_df = df[['left_name', 'right_name', 'feature_difference']]
# pr_df['diff_left_outer'] = df['right_on_left_max_ind'] - df['left_min_ind']
# pr_df['diff_right_outer'] = df['left_on_right_max_ind'] - df['right_min_ind']
#
# # pr_df.rename(columns={'distance': 'distance_v1'}, inplace=True)
#
# del_left_outer = max(pr_df['diff_left_outer'])
# del_right_outer = max(pr_df['diff_right_outer'])
#
# pr_df['diff_left_inner'] = df['right_on_left_min_ind'] - df['right_on_left_max_ind']
# pr_df['diff_right_inner'] = df['left_on_right_min_ind'] - df['left_on_right_max_ind']
#
# del_left_inner = max(pr_df['diff_left_inner'])
# del_right_inner = max(pr_df['diff_right_inner'])
#
# pr_df['distance'] = np.log(1 + pr_df['diff_left_outer'] / del_left_outer) + \
#                     np.log(1 + pr_df['diff_right_outer'] / del_right_outer) + \
#                     np.log(1 + pr_df['diff_left_inner'] / del_left_inner) + \
#                     np.log(1 + pr_df['diff_right_inner'] / del_right_inner) + \
#                     df['feature_difference'] / (20 - df['feature_difference'])
# pr_df.loc[pr_df['feature_difference'] == 20, 'distance'] = (float('inf'))
#
# pr_df = pr_df.sort_values('distance', ascending=False)
#
# pr_df = pr_df.round(3)
df = df[[
    'left_name', 'right_name', 'feature_difference', 'diff_left_outer',
    'diff_right_outer', 'diff_left_inner', 'diff_right_inner', 'distance']
]

if not os.path.exists('/generated_datasets/'):
    os.mkdir('/generated_datasets/')
df.to_csv('/datasets/df.csv')
