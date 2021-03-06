## about

This is the repo for otto competiton.

## goal

We want top 10% ~~and money~~!

## use

Python! Forget about fucking R... `scipy`, `numpy`, `pandas`, `scikit-learn`, `xgboost` and `nolearn` should be everything!

We should learn from the [official guide](http://nbviewer.ipython.org/github/ottogroup/kaggle/blob/master/Otto_Group_Competition.ipynb).

## coding style
* **wrap the process into functions** so that we could import everything and do not mess up everything
* use **four spaces** indention in Python code instead of tab and two spaces (VERY important!)
* divide the codes into blocks, global, data preproccessing, build model, predict, etc.
* always add comments to what is unclear, using lower case letters.
* always use variables in `this_is_a_variable` instead of `thisIsAVariable`, unless it is defined by the packages, etc.


## warning
* when the bugs occur, check whether we clean the `*.pyc* files and restart iPython again.
* Everytime we start iPython, we should run the following first:

  ```python
  %load_ext autoreload
  %autoreload 2
  ```

  See [this post](http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython/10472712#10472712) for details.

## git usage
* the files committed to gitlab should always be runnable.
* deal with the conflicts, if any occurs.
