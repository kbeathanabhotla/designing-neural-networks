from com.designingnn.cnn.ModelGenerator import ModelGenerator
import os
from keras.callbacks import TensorBoard

from com.designingnn.core import AppContext

import numpy as np
import cv2

from keras.utils import np_utils
from sklearn.model_selection import train_test_split


class ModelRunner:
    def __init__(self, model_dir, hyper_parameters, state_space_parameters):
        self.model_dir = model_dir
        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters

    def run_one_model(self, model_descr, iteration):
        for learning_rate in self.hyper_parameters.INITIAL_LEARNING_RATES:
            # Reading data
            train_data_folder = os.path.join(
                os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'data'), AppContext.DATASET), 'train')
            test_data_folder = os.path.join(
                os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'data'), AppContext.DATASET), 'test')

            x_train, y_train, x_test, y_test = self.data_loader(train_data_folder, test_data_folder)

            input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
            # forcing the precision of the pixel values to be 32 bit
            X_train = x_train.astype('float32')
            X_test = x_test.astype('float32')
            # normalize inputs from 0-255 to 0-1
            X_train = X_train / 255.
            X_test = X_test / 255.
            # one hot encode outputs using np_utils.to_categorical inbuilt function
            y_train = np_utils.to_categorical(y_train)
            y_test = np_utils.to_categorical(y_test)
            num_classes = y_test.shape[1]

            # Splitting the training data into training and validation
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            model = ModelGenerator().generate_model(model_descr, input_shape, num_classes, learning_rate)

            tensorboard_log_file = AppContext.DATASET + iteration
            tensorboard_folder = os.path.join(os.path.join(AppContext.APP_BASE_PATH, 'tensorboard'),
                                              '{}'.format(tensorboard_log_file))
            tensorboard = TensorBoard(log_dir=tensorboard_folder)

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=self.hyper_parameters.NUM_ITER_TO_TRY_LR,
                                batch_size=self.hyper_parameters.TRAIN_BATCH_SIZE, verbose=2,
                                callbacks=[tensorboard])

            # Final evaluation of the model
            scores = model.evaluate(X_test, y_test, verbose=0)

            test_acc_dict = {}

            epoc = 1
            for accuracy in history.history['acc']:
                test_acc_dict[epoc] = accuracy
                epoc = epoc + 1

            test_acc_dict[epoc] = scores[1]

            return {
                'learning_rate': learning_rate,
                'status': 'SUCCESS',
                'test_accs': test_acc_dict
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


        #         # Execute.
        #         print "Running [%s]" % solver_path
        #         log_file = self.get_log_fname(model_dir, learning_rate, self.hyper_parameters.NUM_ITER_TO_TRY_LR)
        #
        #         # Check log file for existence.
        #         acc = None
        #         if os.path.exists(log_file):
        #             acc_dict = get_test_accuracies_dict(log_file)
        #             # Check if we got the test accuracy for one epoch.
        #             if self.hp.NUM_ITER_TO_TRY_LR in acc_dict:
        #                 acc = acc_dict[self.hp.NUM_ITER_TO_TRY_LR]
        #         if not acc:
        #             acc, acc_dict = run_caffe_return_accuracy(solver_path, log_file, self.hp.CAFFE_ROOT, gpu_to_use=gpu_to_use)
        #
        #         if check_out_of_memory(log_file):
        #             return {'learning_rate': learning_rate,
        #                     'status': 'OUT_OF_MEMORY',
        #                     'test_accs': {}}
        #
        #         if self.hyper_parameters.NUM_ITER_TO_TRY_LR not in acc_dict:
        #             raise Exception("Model training interrupted during first epoch. Crashing!")
        #         snapshot_file = self.get_snapshot_epoch_fname(model_dir, self.hp.NUM_ITER_TO_TRY_LR)
        #
        #         print "Got accuracy [%f]" % acc
        #         if acc > self.hp.ACC_THRESHOLD:
        #             model_dir, solver_path = ModelGen(self.model_dir, self.hp, self.ssp).save_models(model_descr,
        #                                                                                    learning_rate,
        #                                                                                    self.hp.MAX_STEPS)
        #             log_file = self.get_log_fname_complete(model_dir, learning_rate)
        #
        #             # Check if log file exists.
        #             if os.path.exists(log_file):
        #                 last_iter, last_epoch = get_last_test_epoch(log_file)
        #                 if last_iter > 0:
        #                     snapshot_file = self.get_snapshot_epoch_fname(model_dir, last_iter)
        #                     if last_iter == self.hp.MAX_STEPS:
        #                         test_acc_dict = get_test_accuracies_dict(log_file)
        #                         return {'solver_path': solver_path,
        #                                 'accuracy': acc,
        #                                 'learning_rate': learning_rate,
        #                                 'status': 'OLD_MODEL',
        #                                 'test_accs': test_acc_dict}
        #                     print "Will resume from [%d] using [%s]" % (last_iter, snapshot_file)
        #
        #             test_acc_list = run_caffe_from_snapshot(solver_path, log_file, snapshot_file, self.hp.CAFFE_ROOT, gpu_to_use=gpu_to_use)
        #             test_acc_dict = self.get_test_accuracies_dict(log_file)
        #             if self.hp.MAX_STEPS in test_acc_dict:
        #                 return {'learning_rate': learning_rate,
        #                         'status': 'SUCCESS',
        #                         'test_accs': test_acc_dict}
        #             else:
        #                 if check_out_of_memory(log_file):
        #                     return {'learning_rate': learning_rate,
        #                             'status': 'OUT_OF_MEMORY',
        #                             'test_accs': {}}
        #                 raise Exception("Model training interrupted. Crashing!")
        #
        #         return {'learning_rate': 0.1,
        #                 'status': 'FAIL',
        #                 'test_accs': {}}
        #
        #     # Returns log file name for given model dir when polling for various
        # # learning rates.
        # def get_log_fname(self, model_dir, learning_rate, max_iter):
        #     return '%s/%s_%f_%d.txt' % (model_dir, 'log', learning_rate, max_iter)
        #
        #
        # # Returns log file when training for maximum iterations
        # def get_log_fname_complete(self, model_dir, learning_rate):
        #     return '%s/%s_%f_%d.txt' % (model_dir, 'log_complete', learning_rate, self.hp.NUM_ITER_PER_EPOCH_TRAIN)
        #
        #
        # # Returns the snapshot file name given iteration number. By default
        # # returns snapshot after first epoch
        # def get_snapshot_epoch_fname(self, model_dir, max_iter):
        #     return '%s/%s_iter_%d.solverstate' % (model_dir, 'modelsave', max_iter)
