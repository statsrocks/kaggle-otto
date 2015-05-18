from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from hpsklearn import HyperoptEstimator, pca, svc, any_classifier
from hyperopt import tpe

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
    clf = RandomForestClassifier(n_estimators=600, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)
    

    # linear svc
    clf = LinearSVC(C=1.0, verbose=2)
    clf.fit(X_train, y_train)
    prob_pos = clf.decision_function(X_valid)
    prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    y_valid_predicted = prob_pos
    log_loss_mc(y_valid, y_valid_predicted)
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # well, non-linear svc
    clf = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, cache_size=2000, class_weight=None, verbose=True, max_iter=-1)
    clf.fit(X_train, y_train)
    prob_pos = clf.decision_function(X_valid)
    prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    y_valid_predicted = prob_pos
    log_loss_mc(y_valid, y_valid_predicted)
    # http://stackoverflow.com/questions/29873981/error-with-sklearn-calibratedclassifiercv-and-svm
    clf_isotonic = CalibratedClassifierCV(OneVsRestClassifier(clf), cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # non-linear svc using sigmoidal
    # http://stackoverflow.com/questions/29873981/error-with-sklearn-calibratedclassifiercv-and-svm
    # probability=True
    clf = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, cache_size=2000, class_weight=None, verbose=True, max_iter=-1)
    clf.fit(X_train, y_train)
    y_valid_predicted = clf.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # nusvc, wtf?
    clf = NuSVC(nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=2000, verbose=True, max_iter=-1, random_state=None)
    clf.fit(X_train, y_train)
    prob_pos = clf.decision_function(X_valid)
    prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    y_valid_predicted = prob_pos
    log_loss_mc(y_valid, y_valid_predicted)
    # http://stackoverflow.com/questions/29873981/error-with-sklearn-calibratedclassifiercv-and-svm
    clf_isotonic = CalibratedClassifierCV(OneVsRestClassifier(clf), cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # nusvc using sigmoidal?
    clf = NuSVC(nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=2000, verbose=True, max_iter=-1, random_state=None)
    clf.fit(X_train, y_train)
    y_valid_predicted = clf.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # k means
    clf = KNeighborsClassifier(n_neighbors=9, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
    clf.fit(X_train, y_train)
    y_valid_predicted = clf.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_isotonic.fit(X_train, y_train)
    y_valid_predicted = clf_isotonic.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # hyperopt?!
    estim = HyperoptEstimator( classifier=svc('mySVC') )
    estim.fit(X_train, y_train)


    # pca?!
    # http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py
    pca = PCA()
    logistic = LogisticRegression()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    pipe.fit(X_train, y_train)
    y_valid_predicted = pipe.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)

    # pca + svc
    pca = PCA()
    svc = SVC(probability=False, cache_size=1000, verbose=True)
    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
    n_components = [20, 40, 64, 90]
    Cs = np.logspace(-4, 4, 5)
    #gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1]
    gammas = [0.001, 0.005, 0.01, 0.1, 1]
    estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              svc__C=Cs,
                              svc__gamma=gammas), verbose=2)
    estimator.fit(X_train, y_train)
    y_valid_predicted = estimator.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)


    # wow

    from sklearn.preprocessing import MinMaxScaler
    train_size = 0.8
    X_train, X_valid, y_train, y_valid, scaler = load_train_data(train_size=train_size, scale_it=True, square_root_it=False)
    X_test, X_test_ids = load_test_data(scaler=scaler, square_root_it=False)
    full_X_train, _, full_y_train, _, full_scaler = load_train_data(full_train=True, scale_it=True, square_root_it=False)
    X_test_for_full, X_test_ids = load_test_data(scaler=full_scaler, square_root_it=False)

    mm_scaler = MinMaxScaler()
    X_train = mm_scaler.fit_transform(X_train)
    X_valid = mm_scaler.transform(X_valid)

    svc = SVC(probability=False, cache_size=1000, verbose=False)
    gammas = np.exp2([-7, -5, -3, 0, 3, 5, 7])
    Cs = np.exp2([-7, -5, -3, 0, 3, 5, 7])
    pipe = Pipeline(steps=[('svc', svc)])
    estimator = GridSearchCV(pipe,
                         dict(svc__C=Cs,
                              svc__gamma=gammas), verbose=2)
    estimator.fit(X_train, y_train)
    y_valid_predicted = estimator.predict_proba(X_valid)
    log_loss_mc(y_valid, y_valid_predicted)
    #Pipeline(steps=[('svc', SVC(C=8.0, cache_size=1000, class_weight=None, coef0=0.0, degree=3, gamma=8.0, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False))])
