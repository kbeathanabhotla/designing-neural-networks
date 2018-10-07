from keras import regularizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

def string_to_model(string):

    model = Sequential()

    layers = string.split('\n')
    for layer in layers:
        if layer.startswith('conv'):
            pass

