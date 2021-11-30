## Importing required libraries ##
import tensorflow as tf
from tensorflow.keras import backend
from keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import reuters
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt

## Setting seed ##
tf.random.set_seed(1234)

## Loading data ##
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

## If you want to view the index, execute this chunk ##
words = reuters.get_word_index()
print(words)

## Defining function to vectorize data ##
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

## Vectorizing the train and test data ##
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## Confirming the train and test labels shape ##
print(train_labels.shape)
print(train_data.shape)

## Setting label type as float32 ##
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Encoding labels to categorical ##
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
y_train = to_categorical(y_train, 46)
y_test = to_categorical(y_test, 46)

## Train/Test split ##
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)

## Building, compiling, fitting and plotting model in one cell ##
model = models.Sequential()
model.add(layers.Dense(92, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(92, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
history = model.fit(x_train,
                   y_train,
                   epochs = 20,
                   batch_size = 500,
                   validation_data = (x_valid, y_valid))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)
plt.plot(epochs, loss_values, 'g-o', label = 'Training Loss', color='m')
plt.plot(epochs, val_loss_values, 'r-s', label = 'Validation Loss', color='c')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs, acc_values, 'g-o', label = 'Training Accuracy', color='green')
plt.plot(epochs, val_acc_values, 'r-s', label = 'Validation Accuracy', color='darkred')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

