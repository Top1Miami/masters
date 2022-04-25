import pandas as pd


class FileReader:
    class_name = 'class'
    file_format = '.csv'

    @staticmethod
    def read_file(file_name):
        if FileReader.file_format not in file_name:  # skip datasets not in csv format
            return None
        with open(file_name) as file:
            df = pd.read_csv(file)
            labels = df[FileReader.class_name]
            features = df.drop(FileReader.class_name, axis=1)
            return labels, features
