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

# creates feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique() # gets all unique values
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)) # creates column of feature names

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# functions for individual graphs of the datasets
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

# input function: turns the panda dataset into the dataset object
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# creates the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# trains the model
linear_est.train(train_input_fn) # trains input function
result = linear_est.evaluate(eval_input_fn) # gets model stats by evaluating on test data

clear_output()
print(result['accuracy'])
print(result)

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[3])
print(result[3]['probabilities'][1])