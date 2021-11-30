import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

## Loading the data ##
(train_images, train_labels),(test_images, test_labels) = cifar10.load_data()

## Reshape xblock data and normalize ##
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32')/train_images.max()

test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32')/test_images.max()

## Convert to labels to categorical ##
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## Split into validation and train data ##
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

########## Example of API build ############

# Input model
visible = Input(shape=(32,32,3))

# Create Tower 1
conv11 = Conv2D(32, (3,3), padding = 'same', activation='relu')(visible)
pool11 = MaxPooling2D((2, 2), padding = 'same')(conv11)

# Create Tower 2
conv21 = Conv2D(32, (3,3), padding = 'same', activation='relu')(visible)
pool21 = MaxPooling2D((2,2), padding = 'same')(conv21)

# Concatentate
merge = Concatenate(axis=-1)([pool11, pool21])

# Flatten into fully connected layer
flat = Flatten()(merge)

# Hidden connected layer and output
hidden = Dense(32, activation='relu')(flat)
drop = Dropout(0.5)(hidden)
output = Dense(10, activation='softmax')(drop)

model_example = Model(inputs=visible, outputs=output)
# plot graph
plot_model(model_example)

#######################

############# Another example of model architecture ###########
x = layers.Input(shape=(32,32,3))
a = layers.Conv2D(32, 1, activation='relu', padding='same', strides=2)(x)

b = layers.Conv2D(32, 1, activation='relu', padding='same')(x)
b = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(b)

c = layers.AveragePooling2D(3, strides=2, padding='same')(x)
c = layers.Conv2D(32, 3, activation='relu', padding='same')(c)

d = layers.Conv2D(32, 1, activation='relu', padding='same')(x)
d = layers.Conv2D(32, 3, activation='relu', padding='same')(d)
d = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(d)

output = layers.concatenate([a, b, c, d], axis=-1)

model = Model(inputs=x, outputs=output)
plot_model(model)

#######################

############# Another example of model architecture ###########
x = layers.Input(shape=(32,32,3))
a = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
a = layers.Conv2D(32, 3, activation='relu', padding='same')(a)
b = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
b = layers.Conv2D(32, 3, activation='relu', padding='same')(a)
c = layers.BatchNormalization()(a, b)
d = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
d = layers.Conv2D(32, 3, activation='relu', padding='same')(c)
e = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(x)
e = layers.AveragePooling2D(3, strides=2, padding='valid')(c)
e = layers.Conv2D(32, 3, activation='relu', padding='same')(c)
f = layers.Conv2D(32, 1, activation='relu', padding='same', strides=2)(x)
f = layers.Conv2D(32, 3, activation='relu', padding='same')(c)
f = layers.Conv2D(32, 3, activation='relu', padding='same')(c)
g = layers.concatenate([a, b, c, d, e, f], axis=-1)
h = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(g)
i = layers.BatchNormalization()(h)
j = layers.Flatten()(i)
k = layers.BatchNormalization()(j)
l = layers.Dense(32, activation='relu')(k)
m = layers.Dropout(0.5)(l)
n = layers.Dense(64, activation='softmax')(m)
output = (a, b, c, d, e, f, g, h, i, j, k, l, m, n)
model = Model(inputs=x, outputs=output)
plot_model(model)

#####################

########## Compiling, fitting and plotting model in one cell ##########
backend.clear_session()
np.random.seed(42)
(train_images, train_labels),(test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32')/train_images.max()
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32')/test_images.max()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)  

model.compile(optimizer= "adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

test_scores = model.evaluate(x_test, y_train, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

########## A well optimized model for this data set ###########
input = Input(shape=(32, 32, 3))

x = layers.Conv2D(32, (3,3), activation='relu',padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), activation='relu',padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(128, (5,5), activation='relu',padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (5,5), activation='relu',padding='same')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(512, (3,3), activation='relu',padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, (5,5), activation='relu',padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x= layers.Dense(10)(x)
output= tf.keras.activations.softmax(x)

model = Model(inputs=input, outputs=output,name="Teasdale_Functional_API")
plot_model(model)

########### Again, compiling, fitting and plotting all in one cell ##########

backend.clear_session()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
history = model.fit(train_X, train_y.flatten(), batch_size=64, epochs=30, validation_data=(val_X,val_y.flatten()), callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights = True)])
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
print("Evaluate on test data")
results = model.evaluate(Val_X, val_y, batch_size=128)
print("test loss, test acc:", results)

#####################