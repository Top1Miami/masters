import numpy as np
from ITMO_FS import UnivariateFilter
from ITMO_FS import anova
from ITMO_FS import fechner_corr
from ITMO_FS import spearman_corr

from experiment import PMeLiFComparisonExperiment
from pipeline import PipeLine

pipeline = PipeLine('../datasets')
filters = [UnivariateFilter(spearman_corr),
           UnivariateFilter(fechner_corr),
           UnivariateFilter(anova)]
points = np.random.random_sample(size=(50, len(filters)))

for i in range(0, len(filters)):
    point = np.zeros(len(filters))
    point[i] = 1
    points = np.vstack((points, point))
points = np.vstack((points, np.zeros(len(filters))))
points = np.vstack((points, np.ones(len(filters))))
experiment = PMeLiFComparisonExperiment(5, 'pearson', filters, 10, '../results/{0}/pmelif_plots/', points=points,
                                        delta=0.1)
pipeline.run(experiment)
