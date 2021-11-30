## Importing required libraries and setting seeds ##
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import reuters
from tensorflow.keras import backend
tf.random.set_seed(1)
np.random.seed(1)

## Loading data ##
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

## Defining a function to vectorize data ##
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

## Vectorizing the data ##
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

## Setting labels to float32 ##
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

## Encoding labels to categorical ##
from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
y_train = to_categorical(y_train, 46)
y_test = to_categorical(y_test, 46)

## Training/Validation split ##
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, shuffle= True)

## Building, fitting and plotting model in a single cell ##
backend.clear_session()
model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(l2 = 0.001), input_shape = (10000,)))
model.add(layers.Dense(256, kernel_regularizer = regularizers.l2(l2 = 0.001), activation = 'relu'))
model.add(layers.Dense(46, activation = 'sigmoid'))
model.add(layers.Dropout(0.005))
model.compile(optimizer='rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x_train,
                   y_train,
                   epochs = 20,
                   batch_size = 50,
                   validation_data = (x_val, y_val),
                   callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights = True)])

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, color = 'blue',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'blue', markersize = 5, label = 'Training loss')
plt.plot(epochs, val_loss_values, color = 'magenta',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'magenta', markersize = 5, label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, color = 'green',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'green', markersize = 5, label = 'Training accuracy')
plt.plot(epochs, val_acc_values, color = 'red',
         linestyle = 'solid', marker = 'o',
         markerfacecolor = 'red', markersize = 5, label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(results)