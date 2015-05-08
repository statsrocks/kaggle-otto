from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet

from ggplot import *

from otto_global import logloss_mc, load_train_data, load_test_data, df_to_csv, float32, int32


def sample_dnn_model(X_train, X_valid, y_train, y_valid, max_epochs=20):
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dropout', DropoutLayer),
               ('dense1', DenseLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=200,
                 dropout_p=0.5,
                 dense1_num_units=200,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=max_epochs)

    dnn_model = net0.fit(X_train, y_train)
    return dnn_model


def predict_from_dnn_model(dnn_model, X_test, X_test_ids=None, get_df=True):
    pred = dnn_model.predict_proba(X_test)
    if not get_df:
        return pred
    if X_test_ids is None:
        nrow_test = pred.shape[0]
        X_test_ids = np.array(range(1, nrow_test+1))
    df = pd.concat([pd.Series(X_test_ids), pd.DataFrame(pred)], axis=1)
    df.columns = ['id','Class_1','Class_2','Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    return df


def parse_dnn_model(dnn_model, plot_it=True):
    """
    Interpred and plot the chaning in dnn_model
    """
    df = pd.DataFrame(dnn_model.train_history_)
    if plot_it:
        ggplot(aes(x='epoch', y='valid_loss'), data=df) + geom_line()
    return df


def main():
    train_size = 0.8
    num_round = 250

    X_train, _, y_train, _, scaler = load_train_data(full_train=True, scale_it=True)
    X_train, y_train = float32(X_train), int32(y_train)
    X_test, X_test_ids = load_test_data(scaler=scaler)
    X_test = float32(X_test)
    dnn_model = sample_dnn_model(X_train, None, y_train, None, max_epochs=num_round)
    pred = predict_from_dnn_model(dnn_model, X_test)
    df_to_csv(pred, 'oh-dnn-submission.csv')


if __name__ == '__main__':
    main()
