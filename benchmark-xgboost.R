# https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/12947/achieve-0-50776-on-the-leaderboard-in-a-minute-with-xgboost

# hints:

# 0.4603 using raw data only.
# 0.445
# 0.43977 -Rafi

# decrease number of iterations
# decrease max_depth
# decrease row/column_subsample
# decrease shrinkage factor ('eta' in R)
# However, if you are decreasing some parameters you should increase some other :D E.g. decreasing the shrinkage factor results in a much less loss reduction after one iteration which means you should possibly increase max_iterations. 

# Did anyone find a way to update an existing xgb model with additional trees without having to train the model from scratch?
# you can take a look at https://github.com/dmlc/xgboost/blob/master/R-package/demo/boost_from_prediction.R

# I am on 0.44044 with xgboost in R. Are you tuning 'subsample' and 'colsample_bytree' parmeters?

# Hi Rafi, I'm on 0.45745 in R version. Is it fair to what parameters are worth looking at ? I looked at eta, max depth, min child weight, weights and gamma. Are there others worth looking at ?
# I am looking at the exact same ones you're looking at, with the addition of subsample - not sure how much of a difference that is making. I honestly got lucky with my random grid search I think, I would say concentrate on your eta and make sure the number of rounds you take when training your model is large enough to compensate for the change in your eta value. 


# I have used all the combinations in xgboost -- no success beyond 0.45998. Can you please validate if I'm going in the right direction in tuning?
# ETA - 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.007, 0.008, 0.001
# Depth - 5,6,7,8,9,10,11,12
# Sub sample - 0.90 to 100
# Sometimes -- gamma - 1.15, Child width - 1.15
# Are these fine or am I missing any range of values outside what's given above? Help is much appreciated. I expect to have individual best models before ensembling.
# Based on the max_depth, for ETAs I use the following range:
# ETA - 0.3 nround = 60 to 190
# 0.2 nround = 100 to 300
# 0.1 nround = 250 to 800
# 0.05 nround = 500 to 800
# 0.01 nround = 1500 to 3000
# 0.008 nround = 2500 to 3200
# 0.0001 nround = 25000 to 35000
# I'm completely lost now -- may be I'm doing something fundamentally wrong.
# Interesting - I'm using colsample_bytree, and min_child_weight as well.

require(xgboost)
require(methods)

train = read.csv('data/train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('data/test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 50
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)
