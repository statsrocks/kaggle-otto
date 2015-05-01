# https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12989/code-for-leaderboard-score-of-0-45612-in-3-minutes
# https://gist.github.com/chrisdubois/6b93a8028f4dc40cab49
# maybe depth of 8 better
# graphlab.boosted_trees_classifier.get_default_options()


# W.r.t. parameter tuning I would be interested in how people in the top tier systematically get to their results. I feel that the majority (including myself) heavily relies on try&error here... 
# I can't speak for the top tier competitors, but I can say how I obtained this benchmark. I wrote a for loop and fit 50 models. For each model, I picked a random set of parameters, e.g. pick a random value for max depth from [6, 8, 10, 12], and so on.
# I looked at the models that appeared to have the best logloss on a heldout set, and used those parameters to train on the whole data set.
# You might be interested in this paper, which I believe provides useful insights even if you're not using deep learning methods.
# Random Search for Hyper-Parameter Optimization
# Also, sklearn has RandomSearchCV sklearn.grid_search.ParameterSampler  for exactly this purpose. 



# Do you not already get a logloss estimate, as shown by your first post on this thread? Therefore negating the need for a hold out set to test on. 
# To be more clear, I am suggesting a for loop over lines 51 and 52 of the code sample for this thread: each loop trains the model using a different 'params' dictionary and we compute a score for the heldout data called `va` (short for validation).


import graphlab as gl
import math
import random

train = gl.SFrame.read_csv('data/train.csv')
test = gl.SFrame.read_csv('data/test.csv')
del train['id']

def make_submission(m, test, filename):
    preds = m.predict_topk(test, output_type='probability', k=9)
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds = preds.sort('id')
    preds.save(filename)

def multiclass_logloss(model, test):
    preds = model.predict_topk(test, output_type='probability', k=9)
    preds = preds.unstack(['class', 'probability'], 'probs').unpack('probs', '')
    preds['id'] = preds['id'].astype(int) + 1
    preds = preds.sort('id')
    preds['target'] = test['target']
    neg_log_loss = 0
    for row in preds:
        label = row['target']
        neg_log_loss += - math.log(row[label])
    return  neg_log_loss / preds.num_rows()

def shuffle(sf):
    sf['_id'] = [random.random() for i in xrange(sf.num_rows())]
    sf = sf.sort('_id')
    del sf['_id']
    return sf

def evaluate_logloss(model, train, valid):
    return {'train_logloss': multiclass_logloss(model, train),
            'valid_logloss': multiclass_logloss(model, valid)}

params = {'target': 'target',
          'max_iterations': 250,
          'max_depth': 10,
          'min_child_weight': 4,
          'row_subsample': .9,
          'min_loss_reduction': 1,
          'column_subsample': .8,
          'validation_set': None}

train = shuffle(train)

# Check performance on internal validation set
tr, va = train.random_split(.8)
m = gl.boosted_trees_classifier.create(tr, **params)
print evaluate_logloss(m, tr, va)

# Make final submission by using full training set
m = gl.boosted_trees_classifier.create(train, **params)
make_submission(m, test, 'submission.csv')