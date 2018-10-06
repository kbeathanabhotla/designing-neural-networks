import argparse

from com.designingnn.core import AppContext
from com.designingnn.service.DesignNeuralNetwork import DesignNeuralNetwork


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', help='Which data set')

    args = parser.parse_args()

    print("""
    Starting the application with the following options
    
    dataset path: {}
    
    """.format(args.dataset_path))

    AppContext.DATASET_PATH = args.dataset_path

    DesignNeuralNetwork().start_app()


if __name__ == '__main__': main()
