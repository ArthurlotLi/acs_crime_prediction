# Variable reduction. expe ts iternum and importancethresh. 
# If running by itself, expects a csv file to be in the same folder
# called complete_iter#.csv. Also exports the product dataframe to a
# csv file called complete_iter#_pruned.csv.
#
# NOTE: This function works by fitting a random forest regressor on
# the dataset and returns a dataset with only the more important
# columns. 
# 
# As such, this function has been deemed !!OBSOLETE!! and is not used
# as part of autodataset.py. 

# The function 

# Ex) python variablereduction.py 2 0.001 property

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np 
import argparse

def reducevariables(dataframe,importancethresh,propertyviolent):
    # GLOBAL VARIABLES
    # Will stay with the database no matter what, does not influence actual variable reduction.
    excluded = ["NAME","property", "violent", "state_x", "state_y", "city", "year_y", "year_x", "stateid","placeid","population"]
    # END GLOBAL VARIABLES 

    print("Initializing reducevariables() with importancethresh "+str(importancethresh)+"!")
    property = None
    if propertyviolent == "property":
        property = True
        Y = dataframe['property']
    elif propertyviolent == "violent":
        property = False
        Y = dataframe['violent']
    else:
        print("WARNING! Unknown argument for propertyviolent!!")
        return

    X = dataframe.drop(excluded, axis=1)
    names = X.columns.values.tolist()
    X = np.nan_to_num(X)

    #print(names)
    print(" Fitting random forest regressor with data...")
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    print("  ...Done!")

    results = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
    acceptedfeatures = []
    countrejected = 0
    for i in range(len(results)):
        featurename = results[i][1]
        featureimportance = results[i][0]

        if(featureimportance >= importancethresh):
            acceptedfeatures.append(featurename)
            print("  Accepted " + featurename + ": " +str(featureimportance) + "!")
        else:
            print("  Rejected " + featurename + ": " +str(featureimportance) + "!")
            countrejected = countrejected + 1
    print(" Parsed all data! Total rejected: " + str(countrejected) + ", Total accepted: " + str(len(acceptedfeatures)) + "!")

    productcolumns = excluded
    for feature in acceptedfeatures:
        productcolumns.append(feature)

    productdataframe = dataframe[productcolumns]
    return productdataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('iternum')
    parser.add_argument('importancethresh')
    parser.add_argument('propertyviolent')
    args = parser.parse_args()

    iternum = args.iternum
    importancethresh= float(args.importancethresh)
    propertyviolent = args.propertyviolent

    productprefix = 'complete_iter'
    productsuffix = '.csv'
    inputname = productprefix + str(iternum)+ productsuffix

    dataframe = pd.read_csv(inputname, delimiter=',',encoding="utf-8-sig")

    productdataframe = reducevariables(dataframe, importancethresh, propertyviolent)

    productprefix = 'complete_iter'
    productsuffix = '_pruned.csv'
    print(" Writing results to "+productprefix + str(iternum)+ productsuffix + "...")
    productdataframe.to_csv(productprefix + str(iternum) + productsuffix, index=False)
    print("...Done! Exiting program now! Goodbye!")