import sys
from argparse import ArgumentParser

import numpy as np
from ITMO_FS import UnivariateFilter

from experiment import PMeLiFExperiment
from pipeline import SingleRunPipeline

parser = ArgumentParser()
parser.add_argument('-f', '--file',
                    dest='filename',
                    help='file to read data from, f.e. file.csv')
parser.add_argument('-ns', '--number_samples',
                    dest='number_samples',
                    help='number of samples to use, f.e. 50')
parser.add_argument('-sp', '--save_path',
                    dest='save_path',
                    help='save path, f.e. ../results/{0}/pmelif_plots/')
parser.add_argument('-b', '--baseline',
                    dest='baseline',
                    help='baseline filter, f.e. pearson',
                    default='pearson')
parser.add_argument('-mf', '--melif_filters', nargs='+',
                    help='melif filters, possible list: GiniIndex, SymmetricUncertainty, SpearmanCorr, PearsonCorr, FechnerCorr'
                         'Chi2, Anova, Relief')
parser.add_argument('-mfs', '--max_features_select',
                    help='max features to select',
                    default=10)
print(sys.argv)

args = parser.parse_args()
file_name = args.filename
numer_samples = int(args.number_samples)
save_path = args.save_path
baseline = args.baseline
melif_filters = args.melif_filters

pipeline = SingleRunPipeline(file_name)
filters = [UnivariateFilter(filter_name) for filter_name in melif_filters]
points = np.random.random_sample(size=(50, len(filters)))

for i in range(0, len(filters)):
    point = np.zeros(len(filters))
    point[i] = 1
    points = np.vstack((points, point))
points = np.vstack((points, np.zeros(len(filters))))
points = np.vstack((points, np.ones(len(filters))))
# example of save_path '../results/{0}/pmelif_plots/'
experiment = PMeLiFExperiment(numer_samples, baseline, filters, 10, save_path, points=points, delta=0.1)
pipeline.run(experiment)
