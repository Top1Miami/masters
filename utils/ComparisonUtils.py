import random

import numpy as np

from models import ClassifierScoring


def convert_to_str(x):
    return ','.join([str(i) for i in x])


def estimator_score(X, y, estimator, selected_features):
    return ClassifierScoring('f1_macro').measure(X, y, selected_features, estimator, 3)


def feature_prec_wrapper(mapping):
    def feature_prec(predicted_features):
        correct_guess = 0.
        for feature in predicted_features:
            if feature in mapping:
                correct_guess += 1.
        return correct_guess / len(predicted_features)

    return feature_prec


def feature_rec_wrapper(mapping, known_features):
    def feature_rec(predicted_features):
        correct_guess = 0.
        for feature in predicted_features:
            if feature in mapping:
                correct_guess += 1
        return correct_guess / len(known_features)

    return feature_rec


def generate_mapping(features, known_features):
    mapping = {}
    for i in range(0, len(features)):
        if features[i] in known_features:
            mapping[i] = features[i]
    return mapping


def join_features(l1, l2):
    l3 = l1 + l2
    random.shuffle(l3)
    return l3


def split_features(features, train=10):
    train_features = random.sample(features, train)
    test_features = [i for i in features if i not in train_features]
    print(
        'Number of train features requested {0}, number of train features generated {1}, number of test features generate {2}'.format(
            train, len(train_features), len(test_features))
    )
    return train_features, test_features


def generate_sampling_data(labels, subsample_size):
    class_zero = np.where(labels == 0)[0]
    class_first = np.where(labels == 1)[0]
    count_zero = int(float(len(class_zero)) / len(labels) * subsample_size)
    count_first = int(float(len(class_first)) / len(labels) * subsample_size)
    print('Number of objects', len(labels), 'Number of zero class objects', len(class_zero),
          'Number of first class objects', len(class_first))
    print('Number of scaled zero class', count_zero, 'Number of scaled first class', count_first)
    return class_zero, class_first, count_zero, count_first


def generate_sample(class_zero, class_first, count_zero, count_first):
    zero = np.random.choice(class_zero, count_zero)
    first = np.random.choice(class_first, count_first)
    return [*zero, *first]
