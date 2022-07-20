from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from numpy import concatenate
from keras import callbacks
import matplotlib.pyplot as plt
 


def neuralnetworkmodel():
    # had to work on it via google colab because my stupid computer told me i didnt have tensorflow but atleast it's learning LOL
    url = 'https://raw.githubusercontent.com/Artiexli/acscrimeprediction/master/modeltraining/dataset/complete/complete_iterA1_property.csv?token=AHB5Y4F2CP7NE3T5DBQPBFK6L55KE'
    dataset = read_csv(url)
    data = dataset.dropna(axis=1, how='all')

    # convert all categorical data to numerical data 
    number = LabelEncoder()
    data['state_x'] = number.fit_transform(data['state_x'].astype('str'))
    data['state_y'] = number.fit_transform(data['state_y'].astype('str'))
    data['city'] = number.fit_transform(data['city'].astype('str'))
    data['NAME'] = number.fit_transform(data['NAME'].astype('str'))
   # print(data)
    # create training and testing datasets 
    train_data = data.sample(frac=0.8,random_state=0)
    test_data = data.drop(train_data.index)
    print(test_data)
    # print(train_data)
    # extract the predicted label from training and testing dataset 
    train_labels = train_data.pop('property')
    print(train_labels)
    test_labels = test_data.pop('property')
    print(test_labels)

    # training statistics 
    train_stats = train_data.describe()
    # train_stats.pop('property')
    # print(train_stats)
    train_stats = train_stats.transpose()

    norm_train_data = norm(train_data, train_stats)
    norm_test_data = norm(test_data, train_stats)

    model = build_model(train_data)

    model.summary()
    history = model.fit(norm_train_data, train_labels, epochs=2000, batch_size=32, validation_split=0.2)
    print(history)
    # maybe use patience parameter 

    # prediction time
    prediction_testing = model.predict(norm_test_data).flatten()
    print(prediction_testing)

    error = prediction_testing - test_labels
    print(error)

def norm(x, train_stats):
      return (x - train_stats['mean']) / train_stats['std']

def build_model(train_data):
    # define input
    inputA = Input(shape=[len(train_data.keys())])
    # layers of neural network for inputA
    y = Dense(64, activation="relu")(inputA)
    y = Dense(32, activation="relu")(y)
    y = Dense(4, activation="relu")(y)
    z = Dense(2, activation="relu")(y)
    z = Dense(1, activation="linear")(z)
    optimizers= 'adagrad'
    model = Model(inputA, outputs=z)
    model.compile(loss='mse', optimizer=optimizers, metrics=['mae', 'mse'])
    return model

if __name__ == '__main__':
    neuralnetworkmodel()