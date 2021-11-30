## Required Libraries ##
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend
tf.random.set_seed(1234)

## Importing data set ##
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

## Adding labels to columns and viewing data set ##
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

## The vehicle country of origin needs to be encoded ##
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

## Removing NA's ##
dataset = dataset.dropna(axis=0)

# Split into target and source variables ##
x = dataset.loc[:,['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'USA', 'Europe', 'Japan']]
y = dataset.loc[:, ['MPG']]

## Train/Test split ##
x_train, x_test, y_train, y_test = train_test_split(x, y)

## Normalizing the data by subtracting the mean, and dividing by 1 std. dev. ##
x_train = (x_train - np.average(x_train)) / (np.std(x_train))
x_test = (x_test - np.average(x_test)) / (np.std(x_test))
y_train = (y_train - np.average(y_train)) / (np.std(y_train))
y_test = (y_test - np.average(y_test)) / (np.std(y_test))

## Bulding the first model ##
backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (x_train.shape[1],)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1))

## Compiling the model ##
model.compile(optimizer = 'adam', loss  = 'mse', metrics=['mae'])

## Fitting the model using 1000 epochs and a validation split ##
history = model.fit(x_train,
                   y_train,
                   epochs = 1000,
                   batch_size=100,
                   validation_data=(x_valid, y_valid),
                   verbose = 0)

## Viewing the history ##
hist = pd.DataFrame(history.history)
print(hist.tail())

## Plotting history ##

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['mae']
val_acc_values = history_dict['val_mae']
epochs = range(1, len(history_dict['mae']) + 1)

plt.plot(epochs, loss_values, 'r', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.ylim(0,20)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()