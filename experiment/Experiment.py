import abc


class Experiment(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self, features, labels, file_name):
        pass
