from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

import xgboost as xgb

from ggplot import *

from otto_global import load_train_data, load_test_data, df_to_csv, save_variable, load_variable, average_of_list_of_dfs


def hey_xgb_model(X_train, X_valid, y_train, y_valid, 
    xgb_param=None,
    num_round=50,
    num_fold=3, early_stopping_rounds=100, use_cv=False):
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
        bst = xgb.train(params=xgb_param, dtrain=xg_train, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
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


valuable_params = [
    # [1410]  train-mlogloss:0.179429 valid-mlogloss:0.450791
    # full train kaggle 0.44486
    {'colsample_bytree': 0.6, 'nthread': 36.0, 'min_child_weight': 4.0, 'subsample': 0.8, 'eta': 0.04, 'early_stopping_rounds': 100.0, 'num_round': 1411, 'max_depth': 7.0, 'gamma': 0.7},

    # [1641]  train-mlogloss:0.188990 valid-mlogloss:0.450980
    # full train kaggle 0.44551
    {'colsample_bytree': 0.5, 'nthread': 36.0, 'min_child_weight': 4.0, 'subsample': 0.8, 'eta': 0.04, 'early_stopping_rounds': 100.0, 'num_round': 1642, 'max_depth': 7.0, 'gamma': 0.9},

    # 0.450 at [6434]
    # however, 0.50886 on lb...
    {'colsample_bytree': 0.4210893122302564, 'min_child_weight': 2.0, 'subsample': 0.4408433569894518, 'eta': 0.0018541797977871862, 'early_stopping_rounds': 100.0, 'num_round': 50000.0, 'max_depth': 7.0, 'gamma': 0.7315742663464265},

    # [1360]    train-mlogloss:0.192825 valid-mlogloss:0.454689
    # full train kaggle 0.44783
    { 'eta': 0.04, 'gamma': 0.7, 'max_depth': 7,'min_child_weight': 4, 'max_delta_step': 0, 'colsample_bytree': 0.5, 'subsample': 0.5, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': 9, 'nthread': 8, 'early_stopping_rounds':100, 'num_round':1361, 'seed': np.random.randint(0, 2**32) },


    # interesting
    # based on the first two params
    # for the same split train and test, first param:[1317]  train-mlogloss:0.182814 valid-mlogloss:0.463639
    # this param: [1339]  train-mlogloss:0.179700 valid-mlogloss:0.462997
    # when using 'colsample_bytree': 0.62, 'gamma': 0.68 , we get worst
    {'colsample_bytree': 0.61, 'nthread': 36.0, 'min_child_weight': 4.0, 'subsample': 0.8, 'eta': 0.04, 'early_stopping_rounds': 100.0, 'num_round': 2500, 'max_depth': 7.0, 'gamma': 0.69},

    # interesting
    # based on the first two params
    # for the same split train and test, first param:[1317]  train-mlogloss:0.182814 valid-mlogloss:0.463639
    # this param: [1401]  train-mlogloss:0.174516 valid-mlogloss:0.462797
    {'colsample_bytree': 0.605, 'nthread': 36.0, 'min_child_weight': 4.0, 'subsample': 0.8, 'eta': 0.04, 'early_stopping_rounds': 100.0, 'num_round': 2500, 'max_depth': 7.0, 'gamma': 0.695},

    # based on the first two params
    # for the same split train and test, second param: [1571]   train-mlogloss:0.192842 valid-mlogloss:0.451824
    # this param [1897] train-mlogloss:0.172259 valid-mlogloss:0.447899
    # maybe at epcoh 2110 for transformed scaled is better! fuck!
    {'colsample_bytree': 0.5643669584804276, 'min_child_weight': 4.0, 'subsample': 0.8754254004223411, 'eta': 0.028854422103924096, 'early_stopping_rounds': 100.0, 'num_round': 6000.0, 'max_depth': 8.0, 'gamma': 0.8236709847384176}
]

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
