from keras import regularizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model


from com.designingnn.resources import mnist_state_space_parameters, mnist_hyper_parameters
from com.designingnn.rl.Cnn import parse
from com.designingnn.rl.StateStringUtils import StateStringUtils


class ModelGenerator:
    def __init__(self, hyper_parameters, state_space_parameters):
        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters

    def generate_model(self, model_descr, input_shape, learning_rate):
        net_list = parse('net', model_descr)
        layers = StateStringUtils(self.state_space_parameters).convert_model_string_to_states(net_list)

        is_flattened = False
        input_layer_added = False

        model = Sequential()

        for layer in layers:
            print("""{}, {}, {}, {}, {}, {}, {}, {}""".format(
                layer.layer_type,
                layer.layer_depth,
                layer.filter_depth,
                layer.filter_size,
                layer.stride,
                layer.image_size,
                layer.fc_size,
                layer.terminate))

            if layer.terminate == 1:
                print(self.hyper_parameters.NUM_CLASSES)

                if not is_flattened:
                    model.add(Flatten())

                model.add(Dense(self.hyper_parameters.NUM_CLASSES, kernel_initializer='normal', activation='softmax'))

            elif layer.layer_type == 'conv':
                out_depth = layer.filter_depth
                kernel_size = layer.filter_size
                stride = layer.stride

                if not input_layer_added:
                    model.add(Conv2D(out_depth, (kernel_size, kernel_size), strides=(stride, stride),
                                     padding=self.state_space_parameters.conv_padding,
                                     input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(learning_rate)))
                    input_layer_added = True
                else:
                    model.add(Conv2D(out_depth, (kernel_size, kernel_size), strides=(stride, stride),
                                     padding=self.state_space_parameters.conv_padding,
                                     activation='relu', kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(learning_rate)))
                is_flattened = False

            elif layer.layer_type == 'nin':
                out_depth = layer.filter_depth

                if not input_layer_added:
                    model.add(Conv2D(out_depth, (1, 1), input_shape=input_shape, activation='relu',
                                     kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(learning_rate)))
                    input_layer_added = True
                else:
                    model.add(Conv2D(out_depth, (1, 1), activation='relu',
                                     kernel_initializer='glorot_uniform',
                                     kernel_regularizer=regularizers.l2(learning_rate)))

                is_flattened = False

            elif layer.layer_type == 'gap':
                model.add(AveragePooling2D(strides=(1, 1)))
                is_flattened = False

            elif layer.layer_type == 'fc':
                num_output = layer.fc_size

                if not is_flattened:
                    model.add(Flatten())

                if not input_layer_added:
                    model.add(Dense(num_output, input_dim=input_shape, activation='relu'))
                    input_layer_added = True
                else:
                    model.add(Dense(num_output, activation='relu'))

            elif layer.layer_type == 'dropout':
                dropout_ratio = 0.5 * float(layer.filter_depth) / layer.fc_size
                model.add(Dropout(dropout_ratio))

            elif layer.layer_type == 'pool':
                kernel_size = layer.filter_size
                stride = layer.stride

                model.add(MaxPooling2D((kernel_size, kernel_size), strides=(stride, stride), padding='valid'))

                is_flattened = False

        model.summary()

        optimizer = 'adam'

        if self.hyper_parameters.OPTIMIZER == 'Adam':
            optimizer = Adam(lr=learning_rate, beta_1=self.hyper_parameters.MOMENTUM, epsilon=None,
                             decay=self.hyper_parameters.WEIGHT_DECAY_RATE)
        elif self.hyper_parameters.OPTIMIZER == 'SGD':
            optimizer = SGD(lr=learning_rate, momentum=self.hyper_parameters.MOMENTUM,
                            decay=self.hyper_parameters.WEIGHT_DECAY_RATE)

        # parallel_model = multi_gpu_model(model, gpus=4)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


if __name__ == '__main__':
    str1 = '[C(256,5,1), C(256,3,1), D(1,4), C(64,5,1), C(256,1,1), D(2,4), C(512,3,1), C(128,3,1), D(3,4), C(128,1,1), C(512,3,1), D(4,4), C(256,3,1), GAP(10), SM(10)]'
    str2 = '[C(256,1,1), C(64,3,1), D(1,2), C(512,5,1), C(128,1,1), D(2,2), C(256,5,1), GAP(10), SM(10)]'
    str3 = '[C(128,3,1), C(64,1,1), D(1,2), C(64,5,1), C(64,3,1), D(2,2), GAP(10), SM(10)]'

    mg = ModelGenerator(mnist_hyper_parameters, mnist_state_space_parameters)
    # mg.generate_model(str1, (28, 28, 3), 0.01)
    # mg.generate_model(str2, (28, 28, 3), 0.01)
    # mg.generate_model(str3, (28, 28, 3), 0.01)
    mg.generate_model('[GAP(10), SM(10)]', (28, 28, 3), 0.01)
    print("""
    
    """)
