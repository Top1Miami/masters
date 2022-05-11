import os

import pandas as pd
from matplotlib import pyplot as plt

for i in range(1, 7):
    df = pd.read_csv('../results/' + str(i) + '/distance/distance_dump.csv')
    distances = df['distance']
    distances = distances[distances != 0]
    distances = distances[distances != float('inf')]
    plt.plot(range(0, len(distances)), distances, label=str(i))
save_path = '../results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.legend(loc='upper right')
plt.savefig(save_path + 'plot.png')
