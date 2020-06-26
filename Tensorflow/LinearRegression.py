from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

from tensorflow import feature_column

# load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # feed this data into the model
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # data used to evaluate the model
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() # gets all unique values
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # creates column of feature names

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

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

