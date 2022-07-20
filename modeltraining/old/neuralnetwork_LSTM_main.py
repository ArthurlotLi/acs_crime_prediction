# neuralnetwork_LSTM_main.py
# 
# Creates LSTM neural network model using a multiple multivariate time series,
# which is to say the data of all cities present in the dataset. With this data, 
# this neural network will predict solutions for all (crime rate for next year).
#
# Example usage: 
# python3 neuralnetwork_LSTM_main.py 2 property -v

import argparse

import numpy as np
import pandas as pd 
import datetime as dt
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler # scaling values before and after.

# LSTM Neural network and other stuff.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

# Visualization
import matplotlib as mpl
mpl.use('tkagg') # Need to put these imports before matplotlib because of a glitch with macos. 
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Testing model
import random

def neuralnetwork(iteration, solution, verbose, plot):
  print("\nInitializing neuralnetwork() with iteration " + iteration + " for solution " + solution + ".")

  filename = 'complete_iterA' + iteration + '_' + solution + '.csv'

  # Read in data, use column 6 for the index (year_y).
  if verbose:
    print("Attempting to read " +filename+"...")
  originalData = pd.read_csv('complete/' + filename, index_col = 6) 
  if verbose:
    print("  ...Done!")

  # Sanity checks
  if (solution != 'violent') and (solution != 'property'):
    print("ERROR: solution is not violent or property. Closing...")
    return

  #
  # 1. Data Pre-Processing
  #
  if verbose:
    print("\n*\n* Beginning Data Pre-Processing.\n*")

  # Retain originalData in a separate variable. 
  data = originalData

  # Extract dates (years) to be used in visualization.
  datelist_train = list(data['year_x'])
  datelist_train = list(map(str, datelist_train))
  datelist_train = [dt.datetime.strptime(date, '%Y').date() for date in datelist_train]

  # Map all placeids to an indexed array so we can iterate through them. 
  place_ids = []
  for index, row in data.iterrows():
    if row['placeid'] not in place_ids:
      place_ids.append(row['placeid'])
  if(verbose):
    print("List of seen placeids:")
    print(place_ids)

  # Map all year_x values to an indexed array so we can iterate through them. 
  data_years = []
  for index, row in data.iterrows():
    if row['year_x'] not in data_years:
      data_years.append(row['year_x'])
  if(verbose):
    print("List of seen years:")
    print(data_years)

  # Drop unused columns. Note that we now don't drop placeid nor year_x,
  # as we allow the model to learn from these variables.  
  data = data.drop(columns=['city', 'state_y', 'state_x', 'NAME','stateid'])
  data = data.dropna(axis=1, how='all')

  # Data type formatting
  cols = list(data)
  data = data.astype(str)
  data.applymap(lambda x: x.replace(',',''))
  data = data.astype(float)

  if verbose:
    print("Data Preview:")
    print(data)

  # Note: .to_numpy is considered better than .values, both of which are
  # successors to as_matrix().
  training_set = data.to_numpy()

  if verbose:
    print("\nShape of training set == " + format(training_set.shape) + ".")

  # Always use feature scaling with a standard scaler before and after
  # predictions. Create a standard scaler for all independent variables
  # and one standard scaler for the dependent variable that we want to 
  # predict
  sc = StandardScaler()
  training_set_scaled = sc.fit_transform(training_set)
  data_scaled =  pd.DataFrame(training_set_scaled, columns = cols)

  # Create a scaled prediction set with the type of solution. Property
  # should be column 0, and violent column 1. 
  sc_predict = StandardScaler()
  columnSolutionBegin = 0
  columnSolutionEnd = 1
  if(solution == 'violent'):
    columnSolutionBegin = 1
    columnSolutionEnd = 2
  predict_set_scaled = sc_predict.fit_transform(training_set[:,columnSolutionBegin:columnSolutionEnd])

  # 
  # 2. Creating and Training the LSTM Model
  #

  # Tuning parameters that can be tweaked. 
  layer_1_nodes = 100
  layer_2_nodes = 100
  dropout_level = 0.8
  learning_rate = 0.0001
  loss_function = 'mean_squared_error'
  epochs = 1000
  batch_size=64
  validation_split = 0.2
  rlr_patience = 50
  rlr_factor = 0.5
  es_patience = 80 # Note: we don't use early stopping or rlr. 
  es_min_delta = 1e-10
  
  # Number of years to predict into the future and
  # Number of years to use to predict the future. Do not use the total
  # number of the years we have, because we want to have some for testing.
  # These are important parameters to consider tweaking later on. 
  n_future = 1
  n_past = 1

  # For gauging the performance with real data afterwards.
  num_cities = 5
  # End tuning parameters. 

  if verbose:
    print("\n*\n* Initiating Dataset Allocation and Model Training.\n*")
  
  # Creating a data structure with years and 1 output. X refers to your
  # float feature matrix of shape (the design matrix of your training
  # set, with (n_samples, n_features)). y is the float target vector of
  # the sample (n_samples,), the label vector.
  # 
  # In other words, X's are the features you are using as input for the
  # model. Y's are the expected outcomes or labels. x_train and y_train
  # are practically the same dataset, just split for testing. 
  x_train = []
  y_train = []

  print("\nPredicting " + str(n_future) + " years into the future with " + str(n_past) + " years of data.")

  # For each "row" (sequence + solution). There will be totalYears - n_past 
  # rows. Ex) 10 years and n_past is 7, there will be 3 sequences. 
  # Each sequence is independent, so there will be many duplicates
  # entries. 

  #modifiedData = 

  for year_i in range(0, len(data_years)-1):
    # For every year up until the last - 1 
    year = data_years[year_i]
    year_rows = data.loc[data['year_x'] == int(year)]
    for index, row in year_rows.iterrows():
      try:
        # For each row in this year, add to x_train and add next year's 
        # crime value to y_train. 
        placeid = row['placeid']
        solution_row = data.loc[(data['placeid'] == int(placeid)) & (data['year_x'] == int(year)+1)][solution]
        solution_value = solution_row.iloc[0]

        x_train.append([row])
        y_train.append([solution_value])
      except:
        print("FAILED TO FIND SOLUTION FOR ROW:" )
        print(row)
      
  # Convert arrays into numpy arrays. 
  x_train = np.array(x_train)
  y_train = np.array(y_train)

  # TODO: Scale these suckers. 

  if verbose:
    print("x_train shape = " +format(x_train.shape) + ".")
    print(x_train)
    print("y_train shape = " +format(y_train.shape) + ".")
    print(y_train)

  # 
  # 2. Creating and Training the LSTM Model
  #

  # Define a simple neural network model. As always, start building a new
  # neural network by defining all the variables necessary. 
  model = Sequential()

  # Add the 1st LSTM layer
  # Note the importance of defining the input_shape, where the input shape
  # must be defined correctly to take in a dataset of multiple values. 
  if verbose:
    print("Model input_shape = " + str(n_past) + ',' + str(data.shape[1]-1))
  model.add(LSTM(units=layer_1_nodes, return_sequences=True, input_shape=(n_past, data.shape[1])))

  # Add the 2nd LSTM layer
  model.add(LSTM(units=layer_2_nodes, return_sequences=False))

  # Adding Dropout to avoid overfitting
  model.add(Dropout(dropout_level))

  # Output layer, as always, is Dense with 1 output which is our predicted value. 
  model.add(Dense(units=1, activation='linear'))

  # Compiling the Neural Network with all of this, using adam optimizer. 
  model.compile(optimizer = Adam(learning_rate=learning_rate), loss = loss_function)

  if verbose:
    print("\nModel created. Summary:")
    print(model.summary())

  # With the model created, it is time to begin training it. 
  
  # Define early stopping to save time (don't train if nothing's improving.)
  #es = EarlyStopping(monitor='val_loss', min_delta = es_min_delta, patience = es_patience, verbose = verbose)

  # Similarily, start spinning down the learning rate when a plateau has been detected.
  #rlr = ReduceLROnPlateau(monitor='val_loss', factor = rlr_factor, patience = rlr_patience, verbose = verbose)

  # Define checkpointing so that we can revert in time if we end up worse than we were before. 
  mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1,save_best_only=True, save_weights_only=True)

  # Get logs of parameters, just in case.
  tb = TensorBoard('logs')

  # Let's start training!! 
  history = model.fit(x_train, y_train, shuffle=True, epochs=epochs, callbacks=[mcp, tb], validation_split=validation_split, verbose=verbose, batch_size=batch_size)

  best_val_loss = min(history.history['val_loss'])
  best_loss = min(history.history['loss'])
  if verbose:
    print("\nModel training complete. Lowest val_loss: " + str(best_val_loss) + " Lowest loss: " + str(best_loss))

  # TODO
  return 

  # 
  # 3. Making Predictions With The LSTM Model
  #
  if verbose:
    print("\n*\n* Making Predictions With The LSTM Model.\n*")

  # Create a list of sequence of years for predictions. 
  datelist_future = pd.date_range(start=(datelist_train[-1]), periods=n_future, freq='Y').tolist()

  # Conert Pandas Timestamp to DateTime objects. 
  datelist_future_ = []
  for this_timestamp in datelist_future:
      datelist_future_.append(this_timestamp.date())

  datelist_future = datelist_future_

  # Perform predictions. Also perform train predictions to display context. Note 
  # that the predictions from training will only be for years that the model CAN
  # predict, so for example, if it needs 7 years it will only output predictions
  # # for 2017, 2018, and 2019. 
  predictions_future = model.predict(x_train[-n_future-1:]) # Add an additional item so we can get a line instead of dot. 
  predictions_train = model.predict(x_train)

  # Inverse predictions to original measurements using standardScaler.
  y_pred_future = sc_predict.inverse_transform(predictions_future)
  y_pred_train = sc_predict.inverse_transform(predictions_train)

  datelist_future_modified = datelist_future + datelist_train[-1:]

  PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=[solution]).set_index(pd.Series(datelist_future_modified))
  PREDICTIONS_TRAIN = pd.DataFrame(y_pred_train, columns=[solution]).set_index(pd.Series(datelist_train[-len(y_pred_train):]))

  # Visualize the data. To do so, parse training set timestamps
  # for better visualization. 
  data = pd.DataFrame(data, columns = cols)
  data.index = datelist_train
  data.index = pd.to_datetime(data.index)

  if verbose:
    print("Future predictions:")
    print(PREDICTIONS_FUTURE)

    print("Past predictions:")
    print(PREDICTIONS_TRAIN)

  # Set plot size.
  rcParams['figure.figsize'] = 14, 5

  # Plot parameter
  START_DATE_FOR_PLOTTING = datelist_train[0]

  # Plot future predictions as a dot, not a line if its only one entry. 
  if(len(PREDICTIONS_FUTURE) == 1):
    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE[solution], 'ro', color='r', label='Predicted ' + solution.capitalize() + ' Crime')
  else:
    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE[solution], color='r', label='Predicted ' + solution.capitalize() + ' Crime')

  # Plot train predictions as a dot, not a line if its only one entry.
  if(len(PREDICTIONS_TRAIN) == 1):
    plt.plot(PREDICTIONS_TRAIN.index, PREDICTIONS_TRAIN[solution], 'ro', color='orange', label='Train ' + solution.capitalize() + ' Crime')
  else:
    plt.plot(PREDICTIONS_TRAIN.index, PREDICTIONS_TRAIN[solution], color='orange', label='Train ' + solution.capitalize() + ' Crime')

  # Plot our historical data. 
  plt.plot(data.loc[START_DATE_FOR_PLOTTING:].index, data.loc[START_DATE_FOR_PLOTTING:][solution], color='b', label = 'Actual '+ solution.capitalize() +' Crime')

  # Vertical line to better represent start of predictions. 
  plt.axvline(x = max(data.index), color='green', linewidth=2, linestyle='--')

  # Grid to make visualization more readable. 
  plt.grid(which='major', color='#cccccc', alpha = 0.5)

  # Add a legend and label. 
  plt.legend(shadow=True)
  plt.title('Predicted ' + cityName + ','+cityState+' ' + solution.capitalize() + ' Crime', family='Arial',fontsize=12)
  plt.xlabel('Timeline', family='Arial', fontsize=10)
  plt.ylabel('Annual '+solution.capitalize()+' Crime', family = 'Arial', fontsize=10)
  plt.xticks(rotation=45,fontsize=8)

  # Plot!
  if plot:
    plt.show()

  # To gauge effectiveness of model, conduct a series of predictions
  # using other cities and report the mseTotals. 
  mseTotals = []

  # Attempt to predict Union City statistics. 
  #mseTotals.append(attemptPrediction(model, originalData, iteration, solution, '74630', n_future, n_past, verbose, plot))
  # Attempt to predict Poway City statistics. 
  #mseTotals.append(attemptPrediction(model, originalData, iteration, solution, '58520', n_future, n_past, verbose, plot))

  # Given the originalData set, choose a select number of cities that 
  # are within the data set, or until you're out of cities in the 
  # dataset.
  seen_cities = [placeid]
  valid_cities = list(originalData['placeid'])
  for i in range(0, min(num_cities, len(valid_cities) - 1)): # Don't go over total number of cities. 
    valid_id = None
    while valid_id is None:
      random_index = random.randint(0, len(valid_cities)-1) # Sneaky randint in inclusive!
      if valid_cities[random_index] not in seen_cities:
        valid_id = valid_cities[random_index]
        seen_cities.append(valid_id)
    # Got a valid ID. execute prediction and append MSE.
    mseTotals.append(attemptPrediction(model, originalData, iteration, solution, valid_id, n_future, n_past, verbose, plot))

    # Add note to describe model performance (in the console.)
  caption="\n*******************************\n* SUMMARY:\n*******************************\niter: A" + str(iteration) +' #'+str(placeid) + '\nprediction: '+ str(y_pred_future[0][0]) +'\n'
  caption= caption + '\nlayer_1_nodes: ' + str(layer_1_nodes) + '\nlayer_2_nodes: ' + str(layer_2_nodes) + '\ndropout_level: ' + str(dropout_level) + '\nlearning_rate: ' + str(learning_rate)
  caption = caption + '\nloss_function: ' + str(loss_function) + '\nepochs: ' + str(epochs) + '\nbatch_size: ' + str(batch_size) + '\nvalidation_split: ' + str(validation_split) + '\nrlr_patience: ' + str(rlr_patience)
  caption = caption + '\nrlr_factor: ' + str(rlr_factor) + '\nes_patience: ' + str(es_patience) + '\nes_min_delta: ' + str(es_min_delta) + '\nn_future: ' + str(n_future) + '\nn_past: ' + str(n_past) + '\n'
  caption = caption + '\nval_loss: ' + str(best_val_loss) + '\nloss: ' + str(best_loss) + '\nmseAverage (' +str(num_cities)+' cities): ' + str(sum(mseTotals)/len(mseTotals)) + '\n'

  print(caption)

  return

# Given a model, attempt predictions for a new city. Returns the 
# MSE of all predictions. Can be used to determine the effectiveness
# of a model. 
def attemptPrediction(model, originalData, iteration, solution, placeid, n_future, n_past, verbose, plot):
  if verbose:
    print("\nInitializing attemptPrediction() with iteration " + iteration + " for solution " + solution + " for placeid " + placeid + ".")

  # Sanity checks
  if (solution != 'violent') and (solution != 'property'):
    print("ERROR: solution is not violent or property. Closing...")
    return
  if (int(placeid)< 0) or int(placeid) not in originalData.values:
    print("ERROR: placeid must be valid and present in data. Closing...")
    return
  if (model is None):
    print("ERROR: model must be valid. Closing...")
    return

  # Shave off all other time series, leaving only city data behind.
  data = originalData.loc[originalData['placeid'] == int(placeid)]

  firstRow = data.iloc[0]
  cityName = str(firstRow['city'])
  cityState = str(firstRow['state_x'])
  print("Creating multivariate time series forecast for " + solution + " crime in " + cityName + "," + cityState + ".")

  # Extract dates (years) to be used in visualization.
  datelist_train = list(data['year_x'])
  datelist_train = list(map(str, datelist_train))
  datelist_train = [dt.datetime.strptime(date, '%Y').date() for date in datelist_train]

  data = data.drop(columns=['city', 'state_y', 'state_x', 'NAME', 'year_x','stateid','placeid'])
  data = data.dropna(axis=1, how='all')

  # Data formatting
  cols = list(data)
  data = data.astype(str)
  data.applymap(lambda x: x.replace(',',''))
  data = data.astype(float)

  # Note: .to_numpy is considered better than .values, both of which are
  # successors to as_matrix().
  training_set = data.to_numpy()

  if verbose:
    print("Data Preview:")
    print(data)

  sc = StandardScaler()
  training_set_scaled = sc.fit_transform(training_set)

  # Create a scaled prediction set with the type of solution. Property
  # should be column 0, and violent column 1. 
  sc_predict = StandardScaler()
  columnSolutionBegin = 0
  columnSolutionEnd = 1
  if(solution == 'violent'):
    columnSolutionBegin = 1
    columnSolutionEnd = 2
  predict_set_scaled = sc_predict.fit_transform(training_set[:,columnSolutionBegin:columnSolutionEnd])

  x_train = []
  y_train = []

  # Append all years. 
  for i in range(n_past, len(training_set_scaled) - n_future +1):
      x_train.append(training_set_scaled[i - n_past:i, 0:data.shape[1] - 1])
      y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

  # Convert arrays into numpy arrays. 
  x_train = np.array(x_train)
  y_train = np.array(y_train)

  # Create a list of sequence of years for predictions. 
  datelist_future = pd.date_range(start=(datelist_train[-1]), periods=n_future, freq='Y').tolist()

  # Conert Pandas Timestamp to DateTime objects. 
  datelist_future_ = []
  for this_timestamp in datelist_future:
      datelist_future_.append(this_timestamp.date())

  datelist_future = datelist_future_

  # Perform predictions. Also perform train predictions to display context. Note 
  # that the predictions from training will only be for years that the model CAN
  # predict, so for example, if it needs 7 years it will only output predictions
  # # for 2017, 2018, and 2019. 
  predictions_future = model.predict(x_train[-n_future-1:]) # Add an additional item so we can get a line instead of dot. 
  predictions_train = model.predict(x_train)

  # Inverse predictions to original measurements using standardScaler.
  y_pred_future = sc_predict.inverse_transform(predictions_future)
  y_pred_train = sc_predict.inverse_transform(predictions_train)

  datelist_future_modified = datelist_future + datelist_train[-1:]

  PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=[solution]).set_index(pd.Series(datelist_future_modified))
  PREDICTIONS_TRAIN = pd.DataFrame(y_pred_train, columns=[solution]).set_index(pd.Series(datelist_train[-len(y_pred_train):]))

  # Return the MSE of observed vs predicted. We'll use the non-scaled values. 
  n = len(predictions_train)
  y_observed = pd.DataFrame(training_set_scaled, columns=cols)[-n:][solution]
  mse = (sum(y_observed) - sum(predictions_train)) ** 2
  mse = mse/n

  # Visualize the data. To do so, parse training set timestamps
  # for better visualization. 
  data = pd.DataFrame(data, columns = cols)
  data.index = datelist_train
  data.index = pd.to_datetime(data.index)

  if verbose:
    print("Future predictions:")
    print(PREDICTIONS_FUTURE.iloc[0]) # get only the first year. 

    print("Past predictions:")
    print(PREDICTIONS_TRAIN)

  # Set plot size.
  rcParams['figure.figsize'] = 14, 5

  # Plot parameter
  START_DATE_FOR_PLOTTING = datelist_train[0]

  # Plot future predictions as a dot, not a line if its only one entry. 
  if(len(PREDICTIONS_FUTURE) == 1):
    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE[solution], 'ro', color='r', label='Predicted ' + solution.capitalize() + ' Crime')
  else:
    plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE[solution], color='r', label='Predicted ' + solution.capitalize() + ' Crime')

  # Plot train predictions as a dot, not a line if its only one entry.
  if(len(PREDICTIONS_TRAIN) == 1):
    plt.plot(PREDICTIONS_TRAIN.index, PREDICTIONS_TRAIN[solution], 'ro', color='r')
  else:
    plt.plot(PREDICTIONS_TRAIN.index, PREDICTIONS_TRAIN[solution], color='r')

  # Plot our historical data. 
  plt.plot(data.loc[START_DATE_FOR_PLOTTING:].index, data.loc[START_DATE_FOR_PLOTTING:][solution], color='b', label = 'Actual '+ solution.capitalize() +' Crime')

  # Vertical line to better represent start of predictions. 
  plt.axvline(x = max(data.index), color='green', linewidth=2, linestyle='--')

  # Grid to make visualization more readable. 
  plt.grid(which='major', color='#cccccc', alpha = 0.5)

  # Add a legend and label. 
  plt.legend(shadow=True)
  plt.title('Predicted ' + cityName + ','+cityState+' ' + solution.capitalize() + ' Crime', family='Arial',fontsize=12)
  plt.xlabel('Timeline', family='Arial', fontsize=10)
  plt.ylabel('Annual '+solution.capitalize()+' Crime', family = 'Arial', fontsize=10)
  plt.xticks(rotation=45,fontsize=8)

  # Print MSE calculations at the end. 
  if verbose:
    print("\nPredicted: ")
    print(predictions_train)
    print("Observed: ")
    print(y_observed)
    print("\nMSE = (" +str(sum(y_observed)) + " - " +str(sum(predictions_train))+ ")^2 * 1/"+str(n)+" = " + str(mse))

  # Plot! 
  if plot:
    plt.show()

  return mse[0]


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('iteration')
  parser.add_argument('solution')
  parser.add_argument('-v', action='store_true', default=False)
  parser.add_argument('-p', action='store_true', default=False)
  args = parser.parse_args()

  interation = args.iteration
  solution = args.solution
  verbose = args.v
  plot = args.p

  results = neuralnetwork(interation, solution, verbose, plot)