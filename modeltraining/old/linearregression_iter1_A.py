# A simple linear regression model using
# complete_iter1.csv. This is one of two
# variants, created by Arthur. 

import pandas as pd
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

datacsv = 'complete_iter1.csv'

def linearregression():
    print("Initializing linearregression() for iteration 1!")
    print(" Attempting to read " +datacsv+"...")
    data = pd.read_csv('dataset/complete/' + datacsv, index_col = 0) # complete csv file
    print("  ...Done!")
    datastuff = data.copy()

    datastuff = datastuff.drop(columns=['property'])
    datastuff  = datastuff.drop(columns=['city'])
    datastuff  = datastuff.drop(columns=['state_y'])
    datastuff  = datastuff.drop(columns=['state_x'])
    datastuff  = datastuff.dropna(axis=1, how='all')

    #X = data.values.reshape(-1,1)  # values converts it into a numpy array
    #Y = target.values.reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column

    #X = datastuff
    X = data[['DP03_0002E']]
    Y = data[['property']]

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X,Y)
    Y_pred = linear_regressor.predict(X)  # make predictions
    print(" Predictions made!")
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

if __name__ == "__main__":
    linearregression()