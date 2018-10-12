import os
import pickle
import struct
import sys
import tarfile

import numpy as np
import requests
from scipy import io
import argparse

import tensorflow as tf
import cv2

def download_file(url, save_path):
    with open(save_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        print('Downloading file of size {} MB.'.format(str(long(total_length) / (1024 * 1024))))

        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write('\r[{}{}]'.format('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


def download_svhn_dataset(save_dir):
    train_path = os.path.join(save_dir, 'train_32x32.mat')
    test_path = os.path.join(save_dir, 'test_32x32.mat')

    # print 'Downloading Svhn Train...'
    # download_file('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', train_path)
    #
    # print 'Downloading Svhn Test...'
    # download_file('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', test_path)

    train = io.loadmat(train_path)
    x_train = train['X']
    y_train = train['y']
    del train

    test = io.loadmat(test_path)
    x_test = test['X']
    y_test = test['y']
    del test

    x_train = np.transpose(x_train, (3, 2, 0, 1))
    y_train = y_train.reshape(y_train.shape[:1]) - 1

    x_test = np.transpose(x_test, (3, 2, 0, 1))
    y_test = y_test.reshape(y_test.shape[:1]) - 1

    print("""
        x_train shape   :   {}
        y_train shape   :   {}
        x_test shape    :   {}
        x_test shape    :   {}
        """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


def create_mnist_dataset(save_dir):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("""
        MNIST dataset summary
    
        x_train shape   :   {}
        y_train shape   :   {}
        x_test shape    :   {}
        x_test shape    :   {}
        """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    file_num = 0

    for x, y in zip(x_train, y_train):
        current_file_path = os.path.join(os.path.join(save_dir, 'train'), str(y))

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        cv2.imwrite('{}.png'.format(os.path.join(current_file_path, str(file_num))), x)
        file_num = file_num + 1

    print('Saved training dataset. Started saving test dataset.')

    for x, y in zip(x_test, y_test):
        current_file_path = os.path.join(os.path.join(save_dir, 'test'), str(y))

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        cv2.imwrite('{}.png'.format(os.path.join(current_file_path, str(file_num))), x)
        file_num = file_num + 1

    print('Saved test dataset.')


def download_cifar10_dataset(save_dir):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print("""
            CIFAR10 dataset summary

            x_train shape   :   {}
            y_train shape   :   {}
            x_test shape    :   {}
            x_test shape    :   {}
            """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    file_num = 0

    for x, y in zip(x_train, y_train):
        current_file_path = os.path.join(os.path.join(save_dir, 'train'), str(y))

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        cv2.imwrite('{}.png'.format(os.path.join(current_file_path, str(file_num))), x)
        file_num = file_num + 1

    print('Saved training dataset. Started saving test dataset.')

    for x, y in zip(x_test, y_test):
        current_file_path = os.path.join(os.path.join(save_dir, 'test'), str(y))

        if not os.path.exists(current_file_path):
            os.makedirs(current_file_path)

        cv2.imwrite('{}.png'.format(os.path.join(current_file_path, str(file_num))), x)
        file_num = file_num + 1

    print('Saved test dataset.')

#
# def save_dataset_as_images(x_train, y_train, x_test, y_test, save_dir):
#     train_dir = os.path.join(save_dir, 'training')
#     test_dir = os.path.join(save_dir, 'test')
#
#     print 'Labels train', np.unique(y_train)
#     print 'Labels test', np.unique(y_test)
#
#     x_train = x_train.astype(float)
#     y_train = y_train.astype(float)
#     x_test = x_test.astype(float)
#     y_test = y_test.astype(float)
#
#
# def delete_irrelavent_files(save_dir):
#     pass


def create_dataset(dataset_name, save_dir):
    if dataset_name == 'cifar10':
        download_cifar10_dataset(save_dir)
    elif dataset_name == 'svhn':
        download_svhn_dataset(save_dir)
    elif dataset_name == 'mnist':
        create_mnist_dataset(save_dir)
    else:
        print('Cannot download the dataset \'{}\', please dowwnload it manually!')

    print("Creating dataset completed!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataset_options = ['cifar10', 'svhn', 'mnist']

    parser.add_argument('-dataset', choices=dataset_options, help='Which data set')
    parser.add_argument('-save_folder', help='Where to save images')

    args = parser.parse_args()

    save_folder = os.path.join(args.save_folder, args.dataset)

    os.system('rm -rf "{}/train"'.format(save_folder))
    os.system('rm -rf "{}/test"'.format(save_folder))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    create_dataset(args.dataset, save_folder)


# import tensorflow as tf
# import cv2
#
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# print("""
#     x_train shape   :   {}
#     y_train shape   :   {}
#     x_test shape    :   {}
#     x_test shape    :   {}
#     """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
#
# file_num = 0
# for x,y in zip(x_train,y_train):
#     print(x.shape)
#     print(y)
#     cv2.imwrite('{}_{}.png'.format(y,file_num), x)
#     file_num = file_num + 1
#
#     break

# import cv2
#
# cv2.imwrite('out.png', data_point)
