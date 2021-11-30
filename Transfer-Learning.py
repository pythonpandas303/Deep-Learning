## Importing Required libraries ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from IPython.display import display 
from PIL import Image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os, shutil 

## Setting seed ##
np.random.seed(42)

## Setting base directory ##
base_dir = 'C:/Users/megal/your_file_path'

Setting the subdirectories ##
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

## Using the cats/dogs data set, further subset directories ##
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

## Confirming directory shapes ##
print(len(os.listdir(train_cats_dir)))
print(len(os.listdir(train_dogs_dir)))
print(len(os.listdir(validation_cats_dir)))
print(len(os.listdir(validation_dogs_dir)))
print(len(os.listdir(test_cats_dir)))
print(len(os.listdir(test_dogs_dir)))


## Augmentation of data ##
datagen = ImageDataGenerator( 
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode='nearest')

## Viewing data after augmentation ##
img = image.load_img(os.path.join(train_cats_dir, os.listdir(train_cats_dir)[1]), target_size=(150,150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0 
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

## Data Generation ##
train_datagen2 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen2 = ImageDataGenerator(rescale=1./255) 

train_generator2 = train_datagen2.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validataion_generator2 = train_datagen2.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

test_generator2 = test_datagen2.flow_from_directory( 
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

## Setting up ResNet 152v2 for our application ##
ResNet = tf.keras.applications.ResNet152V2(
         include_top=False,
         weights="imagenet",
         input_shape=(150, 150, 3),
         classifier_activation="softmax",
)

## Freezing last 10 layers ##
for layer in ResNet.layers[:-10]:
  layer.trainable = False
for layer in ResNet.layers:
  print(layer, layer.trainable)

## Building, compiling, fitting, plotting model all in one cell ##
ResNet_train = models.Sequential()
ResNet_train.add(ResNet)
ResNet_train.add(layers.Flatten())
ResNet_train.add(layers.Dense(512, activation = 'relu'))
ResNet_train.add(layers.Dense(1, activation = 'sigmoid'))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validataion_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

ResNet_train.compile(optimizer = optimizers.Adam(lr=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = ResNet_train.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validataion_generator,
    validation_steps=50)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_accuracy = ResNet_train.evaluate_generator(test_generator, steps = 50)
print('test_acc:', test_accuracy)

## Saving Model ##
ResNet_train.save('cats_and_dogs_small_6.h5')