#
# data_preprocessing.py
#
# Given a generated dataset, execute preprocessing and provide properly
# formatted X and y variables. Utilized by all model files to read and
# process training data. 

import pandas as pd
import numpy as np
from pandas.io.formats.format import return_docstring

class DataPreprocessing:
  # Features that we will not be providing to the model.
  # Note these include both solutions "property" and
  # "violent", because it's more useful to have a model
  # without this knowledge. 
  excluded_columns = ["NAME","property", "violent", "state_x", "state_y", "city", "year_y", "year_x", "stateid","placeid","population"]
  data_folder = "./complete"

  # Read the data and section it off based off the solution column. 
  # Do not include certain columns that aren't useful to the model. 
  #
  # Returns tuple of X and y. 
  def read_process_data(self, iternum, solution):
    print("[INFO] Generating X and y for " + str(iternum) + " - " + str(solution) + ".")

    # Sanity check
    if(solution != "property" and solution != "violent"):
      print("[ERROR] Provided solution is not 'property' or 'violent'. Stopping...")
      return

    data = self.read_data(iternum, solution)
    if data is None:
      return_docstring
    
    # Generate X and y. 
    print("[INFO] Processing X and y...")
    y = data[solution].to_numpy().astype("float")
    data_excluded = data.drop(self.excluded_columns, 1)
    X = np.nan_to_num(data_excluded).astype("float")

    return (X, y)

  def read_data(self, iternum, solution):
    # Sanity check
    if(solution != "property" and solution != "violent"):
      print("[ERROR] Provided solution is not 'property' or 'violent'. Stopping...")
      return

    # Load data from csv file
    dataset_filename = "complete_iterA" + str(iternum) + "_" + solution + ".csv"
    dataset_path = self.data_folder + "/" + dataset_filename
    data = pd.read_csv(dataset_path)
    if data is None:
      print("[ERROR] Unable to read " + dataset_path + "! Stopping...")
      return
    print("[INFO] Data read from " + dataset_path + " successfully.")
    return data