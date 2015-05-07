from __future__ import print_function, division, with_statement

import numpy as np
import scipy as sp
import pandas as pd

from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet

from otto_global import logloss_mc, load_train_data, load_test_data
