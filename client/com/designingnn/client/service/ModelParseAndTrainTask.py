import ConfigParser
import json
import os
import threading
import time
import traceback

import cv2
import numpy as np
import requests
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from com.designingnn.client.core import AppContext
from com.designingnn.client.service.KerasEpochCallback import KerasEpochCallback
from com.designingnn.client.service.ModelParser import ModelParser
from com.designingnn.client.service.StatusService import StatusService


class ModelParseAndTrainTask(threading.Thread):
    def __init__(self, model_options):
        super(ModelParseAndTrainTask, self).__init__()
        self.model_options = model_options
        self.status_update_service = StatusService()

        p = ConfigParser.ConfigParser()
        p.read(os.path.join(AppContext.DATASET_DIR, 'hyperparameters.ini'))

        dictionary = {}
        for section in p.sections():
            dictionary[section] = {}
            for option in p.options(section):
                dictionary[section][option] = p.get(section, option)

        self.hyper_parameters = dictionary

    def run(self):

        model_train_summary = {
            'model_id': self.model_options['model_id'],
            'model_def': self.model_options['model_def'],
            'test_accuracy': 0,
            'train_final_accuracy': 0,
            'train_best_epoch_accuracy': 0,
            'time_taken': 0,
            'summary': 'yet to start training',
            'status': 'yet_to_train'
        }

        self.update_client_status('training', model_train_summary)

        try:
            start = time.time()

            train_path = os.path.join(AppContext.DATASET_DIR, 'training')
            test_path = os.path.join(AppContext.DATASET_DIR, 'testing')

            x_train, y_train, x_test, y_test = self.read_data(train_path, test_path)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            input_dim = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

            model_parser = ModelParser(self.model_options['model_def'], input_dim)

            if model_parser.is_first_layer_dense():
                print("first layer is dense, flattening")
                num_pixels = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]

                x_train = x_train.reshape(x_train.shape[0], num_pixels)
                x_val = x_val.reshape(x_val.shape[0], num_pixels)
                x_test = x_test.reshape(x_test.shape[0], num_pixels)

                input_dim = num_pixels

            model = model_parser.generate_model(input_dim)

            reporter_callback = KerasEpochCallback(self.model_options['model_id'], (x_test, y_test))

            model_train_summary['status'] = 'training'
            model_train_summary['summary'] = 'training model'

            self.update_client_status('training', model_train_summary)

            history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                                epochs=int(self.hyper_parameters['training_options']['epochs']),
                                batch_size=int(self.hyper_parameters['training_options']['batch_size']),
                                callbacks=[reporter_callback],
                                verbose=2)

            all_accuracies = history.history['acc']

            model_train_summary['train_best_epoch_accuracy'] = max(all_accuracies)
            model_train_summary['train_final_accuracy'] = all_accuracies[-1]

            scores = model.evaluate(x_test, y_test, verbose=0)
            model_train_summary['test_accuracy'] = scores[1]

            end = time.time()
            hours, rem = divmod(end - start, 3600)
            minutes, seconds = divmod(rem, 60)

            model_train_summary["time_taken"] = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
            model_train_summary["summary"] = "training completed"
            model_train_summary["status"] = "completed"

        except:
            stack_trace = traceback.format_exc()

            print stack_trace
            model_train_summary['summary'] = "failed to train model. "
            model_train_summary["status"] = "failed"
            model_train_summary['stack_trace'] = stack_trace

        self.update_client_status('free', model_train_summary)

    def update_client_status(self, status, model_train_summary):
        self.status_update_service.update_client_status(status)
        self.status_update_service.update_model_train_status(model_train_summary)
        self.report_train_status_to_server(model_train_summary)

    def report_train_status_to_server(self, model_train_summary):
        requests.post(url="http://{}:{}/model-train-status".format(AppContext.SERVER_HOST, AppContext.SERVER_PORT),
                      data=json.dumps(model_train_summary))

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
