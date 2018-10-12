from keras import regularizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

class ModelGenerator:
    def __init__(self):
        pass

    def generate_model(self, model_descr, input_shape, num_classes, learning_rate):
        pass

        # ('NUM', re.compile('[0-9]+')),
        # ('CONV', re.compile('C')),  --
        # ('POOL', re.compile('P')),
        # ('SPLIT', re.compile('S')),
        # ('FC', re.compile('FC')),
        # ('DROP', re.compile('D')),
        # ('GLOBALAVE', re.compile('GAP')),
        # ('NIN', re.compile('NIN')),
        # ('BATCHNORM', re.compile('BN')),
        # ('SOFTMAX', re.compile('SM')),




        # model = Sequential()
        #
        # # We will add 2 Convolution layers with 32 filters of 3x3, keeping the padding as same
        # model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', input_shape=input_shape, activation='relu',
        #                  kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.01)))
        # model.add(
        #     Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
        #            kernel_regularizer=regularizers.l2(0.01)))
        # # Pooling the feature map using a 2x2 pool filter
        # model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='valid'))
        # # Adding 2 more Convolutional layers having 64 filters of 3x3
        # model.add(
        #     Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
        #            kernel_regularizer=regularizers.l2(0.01)))
        # model.add(
        #     Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform',
        #            kernel_regularizer=regularizers.l2(0.01)))
        # # Flatten the feature map
        # model.add(Flatten())
        # # Adding FC Layers
        # model.add(Dense(500, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dropout(0.3))
        # # A softmax activation function is used on the output
        # # to turn the outputs into probability-like values and
        # # allow one class of the 10 to be selected as the model's output #prediction.
        # model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
        # # Checking the model summary
        # model.summary()
        # # Compile model
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # return model


if __name__ == '__main__':
    str = '[C(256,5,1), C(256,3,1), D(1,4), C(64,5,1), C(256,1,1), D(2,4), C(512,3,1), C(128,3,1), D(3,4), C(128,1,1), C(512,3,1), D(4,4), C(256,3,1), GAP(10), SM(10)]'
    str2 = '[C(256,1,1), C(64,3,1), D(1,2), C(512,5,1), C(128,1,1), D(2,2), C(256,5,1), GAP(10), SM(10)]'

    mg = ModelGenerator()
    model1 = mg.generate_model(str2, (28, 28, 3), 10, 0.01)

    model1.summary()

    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # return model
