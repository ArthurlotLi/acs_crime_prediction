#
# feature_reduction.py
#
# An alternative to dimensionality reduction in order to eliminate
# variables that are not useful for model training. We use this
# to properly reduce the dimensions of the initial dataframe
# of the merged ACS and UCR data (1314 features).
#
# Note the methods supported are:
# - rfe_tree
# - pearson_selectkbest
#

from sklearn.feature_selection import RFE, SelectKBest, r_regression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import time

class FeatureReduction:
   # These columns will never be dropped.
  excluded_columns = ["NAME","property", "violent", "state_x", "state_y", "city", "year_y", "year_x", "stateid","placeid","population"]
  implemented_methods = ["rfe_tree", "pearson_selectkbest"]

  # Parameters. For simplicity's sake we don't allow these to be
  # configurable, although this can change in the future. 
  selectkbest_k = 15

  # Quick check to see if a provided string is present in the method. 
  def check_method_present(self, method):
    if method not in self.implemented_methods:
      return False
    return True

  # The main method to use. Provided a specified method and data,
  # return a reduced dataset. 
  def reduce_dataset(self, data, solution, method):
    print("[INFO] Executing "+str(method)+" feature reduction for provided " + solution + "data.")

    # Sanity checks.
    if(solution != "property" and solution != "violent"):
      print("[ERROR] FeatureReduction RFE provided solution is not 'property' or 'violent'. Stopping...")
      return None
    if self.check_method_present(method) is not True:
      print("[ERROR] FeatureReduction does not implement method '" + str(method)+ "'! Stopping...")
      return None
    
    # Preprocessing. Keep excluded columns to the side. Address
    # possible NaN values. 
    data_static = None
    try:
      data_static = data[self.excluded_columns]
    except:
      print("[ERROR] FeatureReduction RFE failed as excluded_columns are not all present!")
      return None
    data_variables_only = data.drop(self.excluded_columns, axis=1)
    #names = data_variables_only.columns.values.tolist()
    data_variables_only_numpy = np.nan_to_num(data_variables_only)

    print("[INFO] Fitting using " + method + "...")
    reducer = None

    # For supervised feature reduction methods, X are all of the ACS
    # variables provided, Y is the solution column as specified. 

    # We'll use the array class variable so as to minimize confusion. 
    if method == self.implemented_methods[0]: # rfe_tree
      reducer = RFE(estimator=DecisionTreeRegressor())
    elif method == self.implemented_methods[1]: # pearson_selectkbest
      reducer = SelectKBest(r_regression, k=self.selectkbest_k)
    else:
      print("[ERROR] FeatureReduction reducer was None! Stopping...")
      return None
    
    time_start = time.time()
    data_selected = reducer.fit_transform(data_variables_only_numpy, data_static[solution])
    time_end = time.time()
    time_duration_seconds = time_end - time_start
    print("[INFO] Fit complete. Total duration: %.4f hours (%.0f seconds)." % ((time_duration_seconds/3600),time_duration_seconds))

    # Re-merge static columns with the selected columns. Thanks to 
    # stack overflow comment remarking how to retreive the column 
    # names after data selection with .get_support(). 
    # https://stackoverflow.com/questions/29586323/how-to-retain-column-headers-of-data-frame-after-pre-processing-in-scikit-learn
    data_selected_pandas = pd.DataFrame(data=data_selected, columns=[data_variables_only.columns[i] for i in range(len(data_variables_only.columns)) if reducer.get_support()[i]])
    data_product = data_static.join(data_selected_pandas)

    print("[INFO] Feature reduction complete. Original shape: ", end="")
    print(data.shape, end="")
    print(", Final shape: ", end="")
    print(data_product.shape, end=".\n")

    return data_product