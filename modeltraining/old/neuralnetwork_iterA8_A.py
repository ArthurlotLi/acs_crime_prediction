# A simple neural network model using
# complete_iterA1_property.csv and complete_iterA1_violent.csv. 
# This is one of two variants, created by Arthur. 

# Dataset to be worked on is 8 50000 70000 California 0.13 0.001

import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import numpy as np
#import torch
#import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def neuralnetwork(filename, solution):
    print("Initializing neuralnetwork() for iteration 1!")
    print(" Attempting to read " +filename+"...")
    data = pd.read_csv('dataset/complete/' + filename, index_col = 0) # complete csv file
    print("  ...Done!")

    if solution == 'violent':
      print(' ...Dropping property variable')
      data = data.drop(columns=['city', 'state_y', 'state_x', 'year_y', 'year_x','stateid','placeid','property'])
    elif solution == 'property':
      print(' ...Dropping violent variable')
      data = data.drop(columns=['city', 'state_y', 'state_x', 'year_y', 'year_x','stateid','placeid','violent'])
    else:
      print("WARNING: solution not found! Whoops! Closing...")
      return
    data = data.dropna(axis=1, how='all')

    # All data has been pre-processed!

    # Create the subset dataframe that we'll be using for training/testing the model
    # THIS SHOULD BE CHANGED WITH DATA SELECTION.
    # The following code is heavily made up of code from https://www.tensorflow.org/tutorials/keras/regression.

    # Split data in to train and test
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset = data.drop(train_dataset.index)
    
    print(" Printings stats of training data...")
    train_stats = train_dataset.describe()
    print(train_stats.transpose())
    #train_stats.pop(solution)
    #train_stats = train_stats.transpose()
    #print(train_stats)

    # Split features from labels. 
    print(" Train data:")
    print(train_dataset)
    print(" Test data:")
    print(test_dataset)
    train_labels = train_dataset.pop(solution)
    test_labels = test_dataset.pop(solution)

    # Use this segment if you don't want to normalize data.
    normed_train_data = train_dataset
    normed_test_data = test_dataset

    print(" Building model...")
    model = build_model(normed_train_data)
    model.summary()

    print(" Training Model...")
    # Training the model
    EPOCHS = 1000

    history =  model.fit(
      normed_train_data, train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0,
      callbacks=[tfdocs.modeling.EpochDots()])

    print(" Evaluating Model...")
    results = model.evaluate(normed_test_data, test_labels, verbose=2)

    examplesize = 20

    print(" Printing example batch...")
    example_batch = normed_train_data[:examplesize]
    example_result = model.predict(example_batch)
    print(example_result)
    print(" Actual batch...")
    print(train_labels[:examplesize])

    # And done!
    return results
    

# A better normalization method
def normalizepandas(df):
  return (df-df.mean())/df.std()
  # return (df-df.min())/(df.max()-df.min())

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def build_model(train_dataset):
  af = 'relu'
  model = keras.Sequential([
    layers.Dense(64, activation=af, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=af),
    #layers.Dense(64, activation=af),
    #layers.Dense(64, activation=af),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(optimizer = optimizer, loss = 'mae', metrics = ['mae','mse'])
  return model

if __name__ == "__main__":
  propertycsv = 'complete_iterA8_property.csv'
  violentcsv = 'complete_iterA8_violent.csv'
  results = neuralnetwork(propertycsv, 'violent')
  #neuralnetwork(propertycsv, 'property')