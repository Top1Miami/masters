from experiment import EnvironmentExperiment
from pipeline import PipeLine

pipeline = PipeLine('../datasets')
experiment = EnvironmentExperiment()
pipeline.run(experiment)
