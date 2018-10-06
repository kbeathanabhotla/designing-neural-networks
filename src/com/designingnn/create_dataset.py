import argparse
import os
import pickle
import struct
import sys
import tarfile

import numpy as np
import requests
from scipy import io


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


def get_byte(file_in):
    int_out = ord(file_in.read(1))
    return int_out


def get_int(file_in):
    int_out = struct.unpack('>i', file_in.read(4))[0]
    return int_out


def get_image(file_in, row=28, col=28):
    raw_data = file_in.read(row * col)
    out_image = np.frombuffer(raw_data, np.uint8)
    out_image = out_image.reshape((28, 28))
    return out_image


def load_mnist(image_fname, label_fname):
    with open(image_fname, "rb") as image_file, open(label_fname, "rb") as label_file:
        assert (get_int(image_file) == 2051)
        assert (get_int(label_file) == 2049)

        n_items_label = get_int(label_file)
        n_items = get_int(image_file)
        assert (n_items_label == n_items)
        assert (get_int(image_file) == 28)
        assert (get_int(image_file) == 28)

        Y = []
        X = np.zeros((n_items, 28, 28, 1), dtype=np.uint8)
        print "Reading [%d] items" % n_items
        for i in range(n_items):
            label = get_byte(label_file)
            assert (label <= 9)
            assert (label >= 0)
            Y.append(label)
            X[i, :] = get_image(image_file)
    return X, np.asarray(Y)


def download_mnist_dataset(save_dir):
    mnist_files = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte',
                   't10k-labels-idx1-ubyte']
    out_mnist_files = []

    print 'Downloading MNIST dataset...'
    for fname in mnist_files:
        out_file = os.path.join(save_dir, "%s" % fname)
        tar_path = os.path.join(save_dir, "%s.gz" % fname)
        out_mnist_files.append(out_file)
        download_file("http://yann.lecun.com/exdb/mnist/%s.gz" % fname, tar_path)
        print 'Download Done, Extracting... [%s]' % tar_path
        os.system('gunzip -f "%s"' % tar_path)

    x_train, y_train = load_mnist(out_mnist_files[0], out_mnist_files[1])
    x_test, y_test = load_mnist(out_mnist_files[2], out_mnist_files[3])

    print("""
        x_train shape   :   {}
        y_train shape   :   {}
        x_test shape    :   {}
        x_test shape    :   {}
        """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


def load_cifar(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 32, 32, 3).astype(np.uint8)
        Y = np.array(Y, dtype=np.int64)
        return X, Y


def download_cifar10_dataset(save_dir):
    print('Downloading CIFAR10 dataset...')
    tar_path = os.path.join(save_dir, "cifar-10-python.tar.gz")
    download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', tar_path)
    tar = tarfile.open(tar_path)
    tar.extractall(save_dir)
    tar.close()

    root = os.path.join(save_dir, "cifar-10-batches-py")

    # Training Data
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        x, y = load_cifar(f)
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    # Testing data
    x_test, y_test = load_cifar(os.path.join(root, 'test_batch'))

    print("""
    x_train shape   :   {}
    y_train shape   :   {}
    x_test shape    :   {}
    x_test shape    :   {}
    """.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    print("""
    {} {} {}  {}
    """.format(type(x_train), type(y_train), type(x_test), type(y_test)))

    return x_train, y_train, x_test, y_test


def save_dataset_as_images(x_train, y_train, x_test, y_test, save_dir):
    train_dir = os.path.join(save_dir, 'training')
    test_dir = os.path.join(save_dir, 'test')

    print 'Labels train', np.unique(y_train)
    print 'Labels test', np.unique(y_test)

    x_train = x_train.astype(float)
    y_train = y_train.astype(float)
    x_test = x_test.astype(float)
    y_test = y_test.astype(float)




def delete_irrelavent_files(save_dir):
    pass


def create_dataset(dataset_name, save_dir):
    if dataset_name == 'cifar10':
        x_train, y_train, x_test, y_test = download_cifar10_dataset(save_dir)
    elif dataset_name == 'svhn':
        x_train, y_train, x_test, y_test = download_svhn_dataset(save_dir)
    elif dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = download_mnist_dataset(save_dir)
    else:
        print('Cannot download the dataset \'{}\', please dowwnload it manually!')

    if x_train is not None:
        print('Saving dataset as images')
        save_dataset_as_images(x_train, y_train, x_test, y_test, save_dir)
        delete_irrelavent_files(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataset_options = ['cifar10', 'svhn', 'mnist']

    parser.add_argument('-dataset', choices=dataset_options, help='Which data set')
    parser.add_argument('-save_folder', help='Where to save images')

    args = parser.parse_args()

    save_folder = os.path.join(args.save_folder, args.dataset)

    os.system('rm -rf "{}"'.format(save_folder))
    os.makedirs(save_folder)

    create_dataset(args.dataset, save_folder)
