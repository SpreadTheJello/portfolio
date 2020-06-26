from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

from tensorflow import feature_column

# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # train from
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # tests model
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

def show_age():
    dftrain.age.hist(bins=20)
    plt.show()

def show_sex():
    dftrain.sex.value_counts().plot(kind='barh')
    plt.show()

def show_class():
    dftrain['class'].value_counts().plot(kind='barh')
    plt.show()

def show_percentSurvivedBySex():
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    plt.show()

show_percentSurvivedBySex()