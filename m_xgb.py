from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

import xgboost as xgb

from ggplot import *

from otto_global import load_train_data, load_test_data, df_to_csv, save_variable, load_variable


def hey_xgb_model(X_train, X_valid, y_train, y_valid, 
    xgb_param=None,
    num_round=50,
    num_fold=3, use_cv=False):
    """
    The function returns the fitted xgb model.
    """
    xgb_force_param = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9
    }
    if xgb_param is None:
        xgb_param = xgb_force_param
    for k, v in xgb_force_param.iteritems():
        xgb_param[k] = v

    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_valid = xgb.DMatrix(X_valid, label=y_valid)

    if not (X_valid is None):
        watchlist = [ (xg_train,'train'), (xg_valid, 'valid') ]
    else:
        watchlist = [ (xg_train,'train')]

    if not use_cv:
        bst = xgb.train(params=xgb_param, dtrain=xg_train, num_boost_round=num_round, evals=watchlist)
    elif use_cv:
        bst = xgb.cv(params=xgb_param, dtrain=xg_train, num_boost_round=num_round, nfold=num_fold)
        bst = _parse_bst_cv_result(bst)

    return bst


def predict_from_xgb_model(bst, X_test, X_test_ids=None, get_df=True):
    xg_test = xgb.DMatrix(X_test)
    pred = bst.predict(xg_test)
    if not get_df:
        return pred
    if X_test_ids is None:
        nrow_test = pred.shape[0]
        X_test_ids = np.array(range(1, nrow_test+1))
    df = pd.concat([pd.Series(X_test_ids), pd.DataFrame(pred)], axis=1)
    df.columns = ['id','Class_1','Class_2','Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
    return df


def _parse_bst_cv_result(bst_cv_result, 
    header=['iter_from_zero', 'test_mlogloss', 'test_mlogloss_sd',
        'train_mlogloss', 'train_mlogloss_sd'], return_df=True):
    """
    If using bst_cv_result=xgb.cv(), bst_cv_result would be a list of test instead of the models.
    So this function parse this result, and return a pd.DataFrame.
    """
    results_list = [
        [
            int(result.split('[')[1].split(']')[0]),
            float(result.split(':')[1].split('+')[0]),
            float(result.split(':')[1].split('+')[1].split('\t')[0]),
            float(result.split(':')[2].split('+')[0]),
            float(result.split(':')[2].split('+')[1]),
        ]
        for result in bst_cv_result
    ]
    df = pd.DataFrame(results_list, columns=header)
    if not return_df:
        # return list of list instead
        return results_list
    return df


def main():
    train_size = 0.8
    num_round = 250
    num_fold = 3
    xgb_basic_param = {
        'eta': 0.3, # maybe low
        'gamma': 1.0, # maybe less than 1
        'max_depth': 10, # absolutely less than 10
        'min_child_weight': 4,
        'max_delta_step': 0,
        'colsample_bytree': 0.8,
        'subsample': 0.9,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 9,
        'nthread': 8,
        'seed': np.random.randint(0, 2**32)
    }

    X_train, X_valid, y_train, y_valid, _ = load_train_data(train_size=train_size)
    X_test, X_test_ids = load_test_data()
    bst = hey_xgb_model(X_train, X_valid, y_train, y_valid, xgb_basic_param, num_round, num_fold, use_cv=False)
    pred = predict_from_xgb_model(bst, X_test)
    df_to_csv(pred, 'non-cv-submission.csv')

    full_X_train, _, full_y_train, _, _ = load_train_data(full_train=True)
    X_test, X_test_ids = load_test_data()
    bst_cv_result = hey_xgb_model(full_X_train, None, full_y_train, None, xgb_basic_param, num_round, num_fold, use_cv=True)
    # How to interpret it? We get the best num_round?!
    ggplot(aes(x='iter_from_zero', y='test_mlogloss'), data=bst_cv_result[50:].reset_index()) + geom_line()
    bst = hey_xgb_model(full_X_train, None, full_y_train, None, xgb_basic_param, num_round=91, num_fold=num_fold, use_cv=False)
    pred = predict_from_xgb_model(bst, X_test)
    df_to_csv(pred, 'oh-cv-submission.csv')


if __name__ == '__main__':
    main()
