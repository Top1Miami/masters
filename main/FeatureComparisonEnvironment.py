from experiment import FeatureComparisonExperiment
from pipeline import PipeLine

pipeline = PipeLine('../datasets')
experiment = FeatureComparisonExperiment(generate_plots=True)
pipeline.run(experiment)
