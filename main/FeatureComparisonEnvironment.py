from experiment import FeatureComparisonExperiment
from pipeline import PipeLine

pipeline = PipeLine('../datasets')
experiment = FeatureComparisonExperiment()
pipeline.run(experiment)
