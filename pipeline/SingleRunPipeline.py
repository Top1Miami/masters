import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from pipeline.FileReader import FileReader


class SingleRunPipeline:
    def __init__(self, file):
        self.file = file

    @staticmethod
    def scale_data(data):
        labels, features = data
        labels.astype(dtype=int)
        scaled_labels = np.where(labels == np.min(labels), 0, 1)
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_labels, scaled_features

    def run(self, experiment):
        read = FileReader.read_file(self.file)
        if read is None:
            return
        print(self.file)
        labels, features = SingleRunPipeline.scale_data(read)
        labels, features = shuffle(labels, features)
        return experiment.run(features, labels, self.file)
