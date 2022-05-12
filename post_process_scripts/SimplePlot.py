import os

import pandas as pd
from matplotlib import pyplot as plt

directory = '../datasets/'
for file_name in os.listdir(directory):  # open directory with datasets
    subname = file_name[:-4]
    df = pd.read_csv('../results/' + subname + '/distance/distance_dump.csv')
    distances = df['distance']
    distances = distances[distances != 0]
    distances = distances[distances != float('inf')]
    plt.plot(range(0, len(distances)), distances, label=subname)
save_path = '../results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.legend(loc='upper right')
plt.savefig(save_path + 'distance_difference.png')
