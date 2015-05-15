from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from ggplot import *

from otto_global import log_loss_mc, load_train_data, load_test_data, df_to_csv, float32, int32, save_variable, load_variable, average_of_list_of_dfs



def main():
    train_size = 0.8


    X_train, X_valid, y_train, y_valid, scaler = load_train_data(train_size=train_size, scale_it=True, square_root_it=True)
    X_test, X_test_ids = load_test_data(scaler=scaler, square_root_it=True)

    full_X_train, _, full_y_train, _, full_scaler = load_train_data(full_train=True, scale_it=True, square_root_it=True)
    X_test_for_full, X_test_ids = load_test_data(scaler=full_scaler, square_root_it=True)


    # logistic
    # loss = ~0.6...
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)
    # clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    # clf_isotonic.fit(X_train, y_train)
    # y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    # log_loss_mc(y_valid, y_valid_predicted)
    

    # gnb
    # loss = ~1.6...
    # clf = GaussianNB()
    # clf.fit(X_train, y_train)
    # clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    # clf_isotonic.fit(X_train, y_train)
    # y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    # log_loss_mc(y_valid, y_valid_predicted)
    

    # rf
    # when n_estimators=100, without calibration, loss = ~0.6
    # when n_estimators=100, with calibration, loss = ~0.483
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)
    

    # linear svc
    # clf = LinearSVC(C=1.0)
    # clf.fit(X_train, y_train)
    # clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    # clf_isotonic.fit(X_train, y_train)
    # y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    # log_loss_mc(y_valid, y_valid_predicted)


    # well, non-linear svc
    # clf = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, cache_size=1000, class_weight=None, verbose=True, max_iter=-1)
    # clf.fit(X_train, y_train)
    # y_valid_predicted = clf.predict_proba(X_valid)
    # #clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    # #clf_isotonic.fit(X_train, y_train)
    # #y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    # log_loss_mc(y_valid, y_valid_predicted)
