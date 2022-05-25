import sys
from argparse import ArgumentParser

import numpy as np
from ITMO_FS import UnivariateFilter

from experiment.FourthPipelineExperiment import FourthPipelineExperiment
from pipeline import SingleRunPipeline

# sys.path.insert(1, '/Users/dmitriish/PycharmProjects/masters/')
# sys.path.insert(1, '/nfs/home/dshusharin/masters/')

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
                         'Chi2, Anova, Relief, InformationGain')
parser.add_argument('-ss', '--sample_size',
                    help='sample size',
                    default=100)
parser.add_argument('-as', '--alpha_start',
                    help='alpha_start',
                    default=0.1)
parser.add_argument('-ae', '--alpha_end',
                    help='alpha_end',
                    default=1.1)
parser.add_argument('-ad', '--alpha_delta',
                    help='alpha_delta',
                    default=0.1)

print(sys.argv)

args = parser.parse_args()
file_name = args.filename
number_samples = int(args.number_samples)
save_path = args.save_path
baseline = args.baseline
melif_filters = args.melif_filters
sample_size = int(args.sample_size)
alpha_start = float(args.alpha_start)
alpha_end = float(args.alpha_end)
alpha_delta = float(args.alpha_delta)

pipeline = SingleRunPipeline(file_name)
filters = [UnivariateFilter(filter_name) for filter_name in melif_filters]

point = np.zeros(len(filters))
point[0] = 1
points = point
for i in range(1, len(filters)):
    point = np.zeros(len(filters))
    point[i] = 1
    points = np.vstack((points, point))
points = np.vstack((points, np.zeros(len(filters))))
points = np.vstack((points, np.ones(len(filters))))
points = np.vstack((points, np.random.random_sample(size=(200, len(filters)))))
# example of save_path '../results/{0}/pmelif_plots/'
experiment = FourthPipelineExperiment(number_samples, baseline, filters, save_path, points=points, delta=0.1,
                                      sample_size=sample_size, alpha_start=alpha_start, alpha_end=alpha_end,
                                      alpha_delta=alpha_delta)
# dataset madelon, madeline sample size 100. gina_agnostic, gina sample size 200. gina_prior, bioresponse doesn't matter.
pipeline.run(experiment)
