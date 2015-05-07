"""
This is the global file.
Everything in here should be functions and callable.
"""

from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.cross_validation import train_test_split


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss
    This function is not officially provided by Kaggle, so there is no
    guarantee for its correctness.
    See https://www.kaggle.com/c/otto-group-product-classification-challenge/details/evaluation .
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll


def load_train_data(path=None, train_size=0.8, full_train=False):
    if path is None:
        try:
            # Unix
            df = pd.read_csv('data/train.csv')
        except IOError:
            # Windows
            df = pd.read_csv('data\\train.csv')
    else:
        df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    if not full_train:
        X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
        print('Loaded training data and splited it into training and validating.')
        return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))
    elif full_train:
        X_train, X_valid, y_train, y_valid = X[:, 1:-1], None, X[:, -1], None
        print('Loaded training data.')
        return (X_train.astype(float), X_valid,
            y_train.astype(str), y_valid)


def load_test_data(path=None):
    if path is None:
        try:
            # Unix
            df = pd.read_csv('data/test.csv')
        except IOError:
            # Windows
            df = pd.read_csv('data\\test.csv')
    else:
        df = pd.read_csv(path)
    X = df.values
    X_test, ids = X[:, 1:], X[:, 0]
    print('Loaded testing data.')
    return X_test.astype(float), ids.astype(str)
