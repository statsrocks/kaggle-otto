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

from otto_global import logloss_mc, load_train_data, load_test_data, df_to_csv, float32, int32



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
    model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2,validation_split=1-train_size, show_accuracy=True)

    return model



class MySequential(Sequential):
    """
    After fitting, we can check the val_loss and val_acc in model.train_history_
    """
    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1,
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):
        from keras.models import standardize_y, Progbar, make_batches
        y = standardize_y(y)

        do_validation = False
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val).")
            do_validation = True
            y_val = standardize_y(y_val)
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        else:
            if 0 < validation_split < 1:
                # If a validation split size is given (e.g. validation_split=0.2)
                # then split X into smaller X and X_val,
                # and split y into smaller y and y_val.
                do_validation = True
                split_at = int(len(X) * (1 - validation_split))
                (X, X_val) = (X[0:split_at], X[split_at:])
                (y, y_val) = (y[0:split_at], y[split_at:])
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))

        index_array = np.arange(len(X))
        train_history = {'val_acc':[], 'val_loss':[]}
        for epoch in range(nb_epoch):
            if verbose:
                print('Epoch', epoch)
                progbar = Progbar(target=len(X), verbose=verbose)
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(len(X), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                if shuffle:
                    batch_ids = index_array[batch_start:batch_end]
                else:
                    batch_ids = slice(batch_start, batch_end)
                X_batch = X[batch_ids]
                y_batch = y[batch_ids]

                if show_accuracy:
                    loss, acc = self._train_with_acc(X_batch, y_batch)
                    log_values = [('loss', loss), ('acc.', acc)]
                else:
                    loss = self._train(X_batch, y_batch)
                    log_values = [('loss', loss)]

                # validation
                if do_validation and (batch_index == len(batches) - 1):
                    if show_accuracy:
                        val_loss, val_acc = self.test(X_val, y_val, accuracy=True)
                        log_values += [('val. loss', val_loss), ('val. acc.', val_acc)]
                    else:
                        val_loss = self.test(X_val, y_val)
                        log_values += [('val. loss', val_loss)]

                # logging
                if verbose:
                    progbar.update(batch_end, log_values)

            train_history['val_loss'].append(log_values[2][1].tolist())
            train_history['val_acc'].append(log_values[3][1].tolist())
        self.train_history_ = train_history


def keras_model_1(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.85):
    """
    ~0.5000 at epoch ~350
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = MySequential()
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
    model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2, validation_split=1-train_size, show_accuracy=True)

    return model


def keras_model_2(X_train, y_train, max_epochs=20, batch_size=16, train_size=0.85):
    """
    ~0.4710 at epcoh 1139 (from 0) after 3 hours
    """
    num_classes = len(np.unique(y_train))
    num_features = X_train.shape[1]

    print("Building model...")

    model = MySequential()
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
    model.fit(X, y, nb_epoch=max_epochs, batch_size=batch_size, verbose=2, validation_split=1-train_size, show_accuracy=True)

    return model


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

    keras_model = sample_keras_model(X_train, y_train, max_epochs=54)
    pred = predict_from_dnn_model(keras_model, X_test)
    df_to_csv(pred, 'oh-keras-submission.csv')

    km_1 = keras_model_1(X_train, y_train, max_epochs=250)
    km_2 = keras_model_2(X_train, y_train, max_epochs=1500)
    


if __name__ == '__main__':
    main()
