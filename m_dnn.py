from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

from ggplot import *

from otto_global import log_loss_mc, load_train_data, load_test_data, df_to_csv, float32, int32, save_variable, load_variable



def sample_lsg_model(X_train, y_train, max_epochs=20):
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


def parse_lsg_model(lsg_model, plot_it=True):
    """
    Interpred and plot the chaning in dnn_model
    """
    df = pd.DataFrame(lsg_model.train_history_)
    if plot_it:
        ggplot(aes(x='epoch', y='valid_loss'), data=df) + geom_line()
    return df



def sample_keras_model(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.8):
    """
    https://github.com/fchollet/keras/blob/master/examples/kaggle_otto_nn.py
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = Sequential()
    model.add(Dense(num_features, 512, init='glorot_uniform'))
    model.add(PReLU((512,)))
    model.add(BatchNormalization((512,)))
    model.add(Dropout(0.5))

    model.add(Dense(512, 512, init='glorot_uniform'))
    model.add(PReLU((512,)))
    model.add(BatchNormalization((512,)))
    model.add(Dropout(0.5))

    model.add(Dense(512, 512, init='glorot_uniform'))
    model.add(PReLU((512,)))
    model.add(BatchNormalization((512,)))
    model.add(Dropout(0.5))

    model.add(Dense(512, num_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")

    print("Training model...")
    X = X_train
    y = np_utils.to_categorical(y_train)
    history = model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2,validation_split=1-train_size, show_accuracy=True)

    return model, history


def keras_model_1(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.85):
    """
    ~0.5000 at epoch ~350
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = Sequential()
    model.add(Dense(num_features, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, num_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.4, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print("Training model...")
    X = X_train
    y = np_utils.to_categorical(y_train)
    history = model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2, validation_split=1-train_size, show_accuracy=True)

    return model, history


def keras_model_2(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.85):
    """
    ~0.4710 at epcoh 1139 (from 0) after 3 hours
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = Sequential()
    model.add(Dropout(0.1))

    model.add(Dense(num_features, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, 400, init='glorot_uniform'))
    model.add(PReLU((400,)))
    model.add(BatchNormalization((400,)))
    model.add(Dropout(0.5))

    model.add(Dense(400, num_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print("Training model...")
    X = X_train
    y = np_utils.to_categorical(y_train)
    history = model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2, validation_split=1-train_size, show_accuracy=True)

    return model, history


def keras_model_oh(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.85):
    """
    ~~
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = Sequential()
    model.add(Dropout(0.05))

    model.add(Dense(num_features, 1024, init='glorot_uniform'))
    model.add(PReLU((1024,)))
    model.add(BatchNormalization((1024,)))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 512, init='glorot_uniform'))
    model.add(PReLU((512,)))
    model.add(BatchNormalization((512,)))
    model.add(Dropout(0.5))

    model.add(Dense(512, 256, init='glorot_uniform'))
    model.add(PReLU((256,)))
    model.add(BatchNormalization((256,)))
    model.add(Dropout(0.5))

    model.add(Dense(256, 128, init='glorot_uniform'))
    model.add(PReLU((128,)))
    model.add(BatchNormalization((128,)))
    model.add(Dropout(0.5))

    model.add(Dense(128, num_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.1, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print("Training model...")
    X = X_train
    y = np_utils.to_categorical(y_train)
    history = model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2, validation_split=1-train_size, show_accuracy=True)

    return model, history

def predict_from_dnn_model(dnn_model, X_test, X_test_ids=None, get_df=True):
    """
    It accepts the result of dnn model and predict the probabilites of test.
    The most awesome thing is that it supports both lsg and keras models! Because they use model.predict_proba() both!
    Returns df if get_df=True, returns matrix if get_df=False .
    """
    pred = dnn_model.predict_proba(X_test)
    if not get_df:
        return pred
    if X_test_ids is None:
        nrow_test = pred.shape[0]
        X_test_ids = np.array(range(1, nrow_test+1))
    df = pd.concat([pd.Series(X_test_ids), pd.DataFrame(pred)], axis=1)
    df.columns = ['id','Class_1','Class_2','Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    return df


def main():
    train_size = 0.8
    num_round = 250

    X_train, _, y_train, _, scaler = load_train_data(full_train=True, scale_it=True)
    X_train, y_train = float32(X_train), int32(y_train)
    X_test, X_test_ids = load_test_data(scaler=scaler)
    X_test = float32(X_test)

    lsg_model = sample_lsg_model(X_train, y_train, max_epochs=num_round)
    pred = predict_from_dnn_model(lsg_model, X_test)
    df_to_csv(pred, 'oh-lsg-submission.csv')

    keras_model, sample_km_history = sample_keras_model(X_train, y_train, max_epochs=54)
    pred = predict_from_dnn_model(keras_model, X_test)
    df_to_csv(pred, 'oh-keras-submission.csv')

    km_1, km_1_history = keras_model_1(X_train, y_train, max_epochs=250)
    km_2, km_2_history = keras_model_2(X_train, y_train, max_epochs=1500)
    km_oh, km_oh_history = keras_model_oh(X_train, y_train, max_epochs=1500)
    


if __name__ == '__main__':
    main()
