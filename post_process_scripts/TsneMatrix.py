import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

pd.set_option('display.max_columns', None)

df = pd.read_csv('../results/madelon/distance/distance_dump.csv')

filter_names = ['su', 'fechner', 'spearman', 'pearson', 'igain', 'gini', 'chi2', 'reliefF', 'anova']
print('Number of filters {0}'.format(len(filter_names)))
index_to_filter_mapping = dict([(i, x) for i, x in enumerate(filter_names)])
filter_to_index_mapping = dict([(x, i) for i, x in enumerate(filter_names)])
filter_graph = np.zeros(shape=(len(filter_names), len(filter_names)))

filter_df = df[['left_name', 'right_name', 'distance']].replace(float('inf'), 10000)
filter_numpy = filter_df.to_numpy()

for filter_info in filter_numpy:
    left = filter_to_index_mapping[filter_info[0]]
    right = filter_to_index_mapping[filter_info[1]]
    filter_graph[left, right] = filter_info[2]
    filter_graph[right, left] = filter_info[2]

# print(filter_graph)
tsne = TSNE(learning_rate='auto', init='random', perplexity=3, early_exaggeration=4, metric='precomputed')
# tsne = TSNE(n_components=gina_prior, learning_rate='auto', init='random', perplexity=madeline, early_exaggeration=10,
#             metric='precomputed')

tsne_results = tsne.fit_transform(filter_graph)
print('Tsne results {0}'.format(tsne_results))

tsne_df = pd.DataFrame()

tsne_df['tsne-2d-one'] = tsne_results[:, 0]
tsne_df['tsne-2d-two'] = tsne_results[:, 1]

# tsne_df['tsne-3d-one'] = tsne_results[:, 0]
# tsne_df['tsne-3d-two'] = tsne_results[:, madelon]
# tsne_df['tsne-3d-three'] = tsne_results[:, gina_agnostic]
tsne_df['hue'] = [index_to_filter_mapping[i] for i in range(0, len(filter_names))]
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=plt.get_cmap('tab10').colors,
    hue='hue',
    data=tsne_df,
    # legend="full"
)
# sns.set_style("whitegrid", {'axes.grid': False})
#
# fig = plt.figure(figsize=(bioresponse, bioresponse))
#
# ax = Axes3D(fig)  # Method madelon
# ax = fig.add_subplot(111, projection='3d') # Method gina_agnostic
#
# sc = ax.scatter(tsne_df['tsne-3d-one'], tsne_df['tsne-3d-two'], tsne_df['tsne-3d-three'], c=tsne_df['tsne-3d-one'],
#                 marker='o', cmap=ListedColormap(sns.color_palette("husl", 256).as_hex()), alpha=madelon)
#
# plt.legend(tsne_df['hue'], bbox_to_anchor=(madelon.05, madelon), loc=gina_agnostic)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

plt.show()
