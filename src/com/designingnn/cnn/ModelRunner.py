import os
import time

import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from com.designingnn.cnn.ModelGenerator import ModelGenerator
from com.designingnn.core import AppContext


class ModelRunner:
    def __init__(self, model_dir, hyper_parameters, state_space_parameters):
        self.model_dir = model_dir
        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters

        train_data_folder = os.path.join(
            os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'data'), AppContext.DATASET), 'train')
        test_data_folder = os.path.join(
            os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'data'), AppContext.DATASET), 'test')

        x_train, y_train, x_test, y_test = self.data_loader(train_data_folder, test_data_folder)

        self.input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        # forcing the precision of the pixel values to be 32 bit
        X_train = x_train.astype('float32')
        X_test = x_test.astype('float32')
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255.
        self.X_test = X_test / 255.
        # one hot encode outputs using np_utils.to_categorical inbuilt function
        y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        num_classes = self.y_test.shape[1]

        # Splitting the training data into training and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        print("read data")

    def run_one_model(self, model_descr, iteration):
        for learning_rate in self.hyper_parameters.INITIAL_LEARNING_RATES:
            # Reading data

            model = ModelGenerator(self.hyper_parameters, self.state_space_parameters).generate_model(model_descr,
                                                                                                      self.input_shape,
                                                                                                      learning_rate)

            tensorboard_log_file = AppContext.DATASET + '_iter_' + str(int(iteration)) + '_' +str(time.time())
            tensorboard_folder = os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'tensorboard'),
                                              '{}'.format(tensorboard_log_file))
            tensorboard = TensorBoard(log_dir=tensorboard_folder)

            history = model.fit(
                self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                epochs=self.hyper_parameters.MAX_EPOCHS,
                batch_size=self.hyper_parameters.TRAIN_BATCH_SIZE,
                verbose=2,
                callbacks=[tensorboard]
            )

            # Final evaluation of the model
            scores = model.evaluate(self.X_test, self.y_test, verbose=0)

            print("""
            test data scores {}
            """.format(str(scores)))

            test_acc_dict = {}

            epoc = 1
            for accuracy in history.history['acc']:
                test_acc_dict[epoc] = accuracy
                epoc = epoc + 1

            # test_acc_dict[epoc] = scores[1]

            return {
                'learning_rate': learning_rate,
                'status': 'SUCCESS',
                'test_accs': test_acc_dict,
                'test_data_accuracy': scores[1]
            }

    def data_loader(self, path_train, path_test):
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
