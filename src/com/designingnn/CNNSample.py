import os

import cv2
import numpy as np
from keras import regularizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
# we always initialize the random number generator to a constant seed #value for reproducibility of results.
seed = 7
np.random.seed(seed)


# load data from the path specified by the user
def data_loader(path_train, path_test):
    train_list = os.listdir(path_train)
    '''
    # Map class names to integer labels
    train_class_labels = { label: index for index, label in enumerate(class_names) } 
    '''
    # Number of classes in the dataset
    num_classes = len(train_list)

    # Empty lists for loading training and testing data images as well as corresponding labels
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    # Loading training data
    for label, elem in enumerate(train_list):

        path1 = path_train + '/' + str(elem)
        images = os.listdir(path1)
        for elem2 in images:
            path2 = path1 + '/' + str(elem2)
            # Read the image form the directory
            img = cv2.imread(path2)
            # Append image to the train data list
            x_train.append(img)
            # Append class-label corresponding to the image
            y_train.append(str(label))

        # Loading testing data
        path1 = path_test + '/' + str(elem)
        images = os.listdir(path1)
        for elem2 in images:
            path2 = path1 + '/' + str(elem2)
            # Read the image form the directory
            img = cv2.imread(path2)
            # Append image to the test data list
            x_test.append(img)
            # Append class-label corresponding to the image
            y_test.append(str(label))

    # Convert lists into numpy arrays
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test


path_train = '/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/data/mnist/train'
path_test = '/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/data/mnist/test'

X_train, y_train, X_test, y_test = data_loader(path_train, path_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.
# one hot encode outputs using np_utils.to_categorical inbuilt function
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Splitting the trining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print "num classes " + str(num_classes)
print "ip shape " + str(input_shape)

# define baseline model
# The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784)
def baseline_model():
    # create model
    model = Sequential()
    # We will add 2 Convolution layers with 32 filters of 3x3, keeping the padding as same
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape, activation='relu',
                     kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(0.01)))
    # Pooling the feature map using a 2x2 pool filter
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))
    # Adding 2 more Convolutional layers having 64 filters of 3x3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(0.01)))
    # Flatten the feature map
    model.add(Flatten())
    # Adding FC Layers
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    # A softmax activation function is used on the output
    # to turn the outputs into probability-like values and
    # allow one class of the 10 to be selected as the model's output #prediction.
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Checking the model summary
    model.summary()
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
# The model is fit over 10 epochs with updates every 200 images. The test data is used as the validation dataset

from time import time
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/tensorboard/cnnSample{}".format(time()))

x = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=200,
    verbose=2,
    callbacks=[tensorboard]
    # steps_per_epoch=5000,
    # validation_steps=5000
)

print x.history
print type(x)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("""
        test data scores {}
        """.format(str(scores)))
print scores
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
