# A simple neural network model using
# complete_iter1.csv. This is one of two
# variants, created by Arthur. 

# Note that this is currently an experimental
# Model - it only uses five select variables!

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


datacsv = 'complete_iter1.csv'

def neuralnetwork():
    print("Initializing neuralnetwork() for iteration 1!")
    print(" Attempting to read " +datacsv+"...")
    datafile = pd.read_csv('dataset/complete/' + datacsv, index_col = 0) # complete csv file
    print("  ...Done!")
    datastuff = datafile.copy()

    #datastuff = datastuff.drop(columns=['property'])
    datastuff  = datastuff.drop(columns=['city'])
    datastuff  = datastuff.drop(columns=['state_y'])
    datastuff  = datastuff.drop(columns=['state_x'])
    datastuff  = datastuff.dropna(axis=1, how='all')

    # All data has been pre-processed!

    #print(datastuff[['NAME']])
    #print(datastuff[['DP03_0002E']])
    #print(type(datastuff[['DP03_0002E']]))
    #print(datastuff[['property']])

    # Create the subset dataframe that we'll be using for training/testing the model
    # THIS SHOULD BE CHANGED WITH DATA SELECTION.
    # The following code is heavily made up of code from https://www.tensorflow.org/tutorials/keras/regression.
    data = datastuff[['DP02_0059E', 'DP02_0064E', 'DP04_0136E', 'DP04_0058E', 'DP03_0076E', 'DP03_0002E', 'property']].copy()

    # Split data in to train and test
    train_dataset = data.sample(frac=0.8, random_state=0)
    test_dataset = data.drop(train_dataset.index)
    sns.pairplot(train_dataset[['DP02_0059E', 'DP02_0064E', 'DP04_0136E', 'DP04_0058E', 'DP03_0076E', 'DP03_0002E', 'property']], diag_kind = "kde")
    
    print(" Printings stats of training data...")
    train_stats = train_dataset.describe()
    train_stats.pop("property")
    train_stats = train_stats.transpose()
    print(train_stats)

    # Split features from labels. 
    train_labels = train_dataset.pop('property')
    test_labels = test_dataset.pop('property')

    # Normalization.
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    # The model.
    model = build_model(train_dataset)
    model.summary()

    # Example batch.
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    example_result

    # Training the model
    EPOCHS = 2000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split = 0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    # Visualize progress
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    # Plot results
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [Property]')

    plotter.plot({'Basic': history}, metric = "mse")
    plt.ylim([0, 20])
    plt.ylabel('MSE [Property^2]')

    # Now use early stop
    model = build_model(train_dataset)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    early_history = model.fit(normed_train_data, train_labels, 
                        epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()])

    plotter.plot({'Early Stopping': early_history}, metric = "mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [Property]')

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} Property".format(mae))

    # Finally, make preditions using test. 
    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Property]')
    plt.ylabel('Predictions [Property]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

    error = test_predictions - test_labels
    plt.hist(error, bins = 25)
    plt.xlabel("Prediction Error [Property]")
    _ = plt.ylabel("Count")

    # And done!

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

if __name__ == "__main__":
    neuralnetwork()