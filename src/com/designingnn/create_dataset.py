import argparse
import os
import requests
import tarfile
import sys
import pickle
import numpy as np

def download_file(url, save_path):
    with open(save_path, "wb") as f:
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        print('Downloading file of size {} MB.'.format(str(long(total_length)/(1024 * 1024))))

        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[{}{}]".format('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def download_svhn_dataset(save_dir):
    pass


def download_mnist_dataset(save_dir):
    pass


def load_cifar(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype(np.uint8)
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

    return x_train, y_train, x_test, y_test


def save_dataset_as_images(dataset, save_dir):
    train_dir = os.path.join(save_dir, 'training')
    test_dir = os.path.join(save_dir, 'test')

    pass


def create_dataset(dataset_name, save_dir):
    dataset = None
    if dataset_name == 'cifar10':
        dataset = download_cifar10_dataset(save_dir)
    elif dataset_name == 'svhn':
        dataset = download_svhn_dataset(save_dir)
    elif dataset_name == 'mnist':
        dataset = download_mnist_dataset(save_dir)
    else:
        print('Cannot download the dataset \'{}\', please dowwnload it manually!')

    if dataset is not None:
        print('Saving dataset as images')
        save_dataset_as_images(dataset, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataset_options = ['cifar10', 'svhn', 'mnist']

    parser.add_argument('-dataset', choices=dataset_options, help='Which data set')
    parser.add_argument('-save_folder', help='Where to save images')

    args = parser.parse_args()

    if not os.path.isdir(args.save_folder):
        print('root directory doesn\'t exist, creating it.')
        os.makedirs(args.save_folder)

    create_dataset(args.dataset, args.save_folder)
