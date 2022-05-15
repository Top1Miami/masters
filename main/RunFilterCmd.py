import sys
from argparse import ArgumentParser

from experiment.FilterComparisonExperiment import FilterComparisonExperiment
from pipeline import SingleRunPipeline

# sys.path.insert(1, '/Users/dmitriish/PycharmProjects/masters/')
# sys.path.insert(1, '/nfs/home/dshusharin/masters/')

parser = ArgumentParser()
parser.add_argument('-f', '--file',
                    dest='filename',
                    help='file to read data from, f.e. file.csv')
parser.add_argument('-ns', '--number_samples',
                    dest='number_samples',
                    help='number of samples to use, f.e. 50',
                    default=100)
parser.add_argument('-sp', '--save_path',
                    dest='save_path',
                    help='save path, f.e. ../results/{0}/pmelif_plots/')
parser.add_argument('-b', '--baseline',
                    dest='baseline',
                    help='baseline filter, f.e. pearson',
                    default='pearson')
parser.add_argument('-mfs', '--max_features_select',
                    help='max features to select',
                    default=10)
parser.add_argument('-ss', '--sample_size',
                    help='sample size',
                    default=100)
print(sys.argv)

args = parser.parse_args()
file_name = args.filename
number_samples = int(args.number_samples)
save_path = args.save_path
baseline = args.baseline
sample_size = int(args.sample_size)

pipeline = SingleRunPipeline(file_name)
# example of save_path '../results/{0}/pmelif_plots/'
experiment = FilterComparisonExperiment(number_samples, baseline, 10, save_path, sample_size=sample_size)
# dataset madelon, madeline sample size 100. gina_agnostic, gina sample size 200. gina_prior, bioresponse doesn't matter.
pipeline.run(experiment)
