import argparse

from com.designingnn.core import AppContext
from com.designingnn.service.DesignNeuralNetwork import DesignNeuralNetwork


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', help='Which data set')

    args = parser.parse_args()

    print("""
    Starting the application with the following options
    
    dataset: {}
    
    """.format(args.dataset))

    AppContext.DATASET = args.dataset

    DesignNeuralNetwork().start_app()


if __name__ == '__main__':
    main()
