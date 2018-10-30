import threading

from com.designingnn.client.service.ModelParser import ModelParser
from com.designingnn.client.service.StatusService import StatusService

from com.designingnn.client.core import AppContext
from com.designingnn.client.service.ModelParser import ModelParser
import numpy as np
import os
from keras.utils import np_utils
import cv2

from sklearn.model_selection import train_test_split


class ModelParseAndTrainTask(threading.Thread):
    def __init__(self, model_options):
        self.model_options = model_options
        self.status_update_service = StatusService()

        StatusService().update_model_training_info()

    def run(self):
        model = ModelParser().parse_model(self.model_options['model_def'])

        self.set_client_status('training')
        try:
            train_path = os.path.join(AppContext.DATASET_DIR, 'training')
            test_path = os.path.join(AppContext.DATASET_DIR, 'testing')

            x_train, y_train, x_test, y_test = self.read_data(train_path, test_path)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            model = ModelParser().parse_model(model_def)
            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=200, verbose=2)
            # Final evaluation of the model
            scores = model.evaluate(x_test, y_test, verbose=0)
        except:
            pass

        self.set_client_status('free')

        pass

    def read_data(self, path_train, path_test):
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

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # normalize inputs from 0-255 to 0-1
        x_train = x_train / 255.
        x_test = x_test / 255.
        # one hot encode outputs using np_utils.to_categorical inbuilt function
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    def set_client_status(self, status):
        if status == 'training':
            if self.status_update_service.get_client_status() == 'free':
                self.status_update_service.update_client_status(status)
            else:
                raise ValueError('Cannot set status for busy client')
        elif status == 'free':
            self.status_update_service.update_client_status(status)
