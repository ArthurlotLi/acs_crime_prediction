#
# dimensionality_reduction.py
#
# Revised dimensionality reduction approach that supersedes 
# variablereduction.py. Utilizes Principal Component Analysis
# to properly reduce the dimensions of the initial dataframe
# of the merged ACS and UCR data (1314 features).

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class DimensionalityReduction:
  # These columns will never be dropped as part of DR. 
  excluded_columns = ["NAME","property", "violent", "state_x", "state_y", "city", "year_y", "year_x", "stateid","placeid","population"]

  # Given data, execute sklearn's PCA decomposition. Allow
  # a number of arguments to be passed through. 
  #
  # Note that given this is Principal Component Analysis, the
  # feature names will not be preserved as the data is 
  # transformed. 
  def principal_component_analysis(self, data, solution):
    print("[INFO] Executing principal component analysis for " + solution + "data.")

    # Sanity check
    if(solution != "property" and solution != "violent"):
      print("[ERROR] DimensionalityReduction PCA provided solution is not 'property' or 'violent'. Stopping...")
      return None

    # Preprocessing. Keep excluded columns to the side. Address
    # possible NaN values. 
    data_static = None
    try:
      data_static = data[self.excluded_columns]
    except:
      print("[ERROR] DimensionalityReduction PCA failed as excluded_columns are not all present!")
      return None
    data_variables_only = data.drop(self.excluded_columns, axis=1)
    #names = data_variables_only.columns.values.tolist()
    data_variables_only = np.nan_to_num(data_variables_only)

    print("[INFO] Fitting...")
    pca = PCA()
    data_selected = pca.fit_transform(data_variables_only)

    # As PCA results in components that are not interpretable, we 
    # no longer have column values. 
    data_product = data_static.join(pd.DataFrame(data=data_selected))

    print("[INFO] Decomposition complete. Final shape: ", end="")
    print(data_product.shape, end=".\n")

    return data_product
