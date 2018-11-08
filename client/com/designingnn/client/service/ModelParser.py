from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.utils import multi_gpu_model

from com.designingnn.client.core import AppContext


class ModelParser:
    def __init__(self):
        pass

    def get_layer(self, layer, input_dim=None, layer_num=None):
        layer_name = layer[0:layer.find("(")]
        print layer_name

        if layer_name == 'SOFTMAX':
            num_out_neurons = int(layer.replace('SOFTMAX', '').replace('(', '').replace(')', '').strip())

            if layer_num != 1:
                return Dense(num_out_neurons, kernel_initializer='normal', activation='softmax')
            else:
                return Dense(num_out_neurons, input_shape=input_dim[0] * input_dim[1] * input_dim[2],
                             kernel_initializer='normal', activation='softmax')

        elif layer_name == 'CONV':
            params = layer.replace('CONV', '').replace('(', '').replace(')', '').strip().split(',')
            num_filters = int(params[0])
            filter_size = int(params[1])
            stride = int(params[2])

            if layer_num != 1:
                return Conv2D(num_filters, (filter_size, filter_size), strides=(stride, stride), padding='same',
                              activation='relu')
            else:
                return Conv2D(num_filters, (filter_size, filter_size), input_shape=input_dim, strides=(stride, stride),
                              padding='same',
                              activation='relu')

        elif layer_name == 'MAXPOOLING':
            pool_size = int(layer.replace('MAXPOOLING', '').replace('(', '').replace(')', '').strip())
            return MaxPooling2D((pool_size, pool_size))

        elif layer_name == 'AVGPOOLING':
            pool_size = int(layer.replace('AVGPOOLING', '').replace('(', '').replace(')', '').strip())
            return AveragePooling2D((pool_size, pool_size))

        elif layer_name == 'DENSE':
            params = layer.replace('DENSE', '').replace('(', '').replace(')', '').strip().split(',')
            num_output = int(params[0])

            if layer_num != 1:
                return Dense(num_output, activation='relu')
            else:
                flattened_value = input_dim[0]*input_dim[1]*input_dim[2]
                print(flattened_value)
                return Dense(num_output, input_dim=flattened_value, activation='relu')

    def generate_model(self, model_def, input_dim):
        model_def = model_def[1:-1]
        print model_def

        should_flatten = True

        model = Sequential()

        layers = model_def.split("),")
        layer_num = 1
        for layer in layers:
            layer = layer.strip()
            if layer[-1] != ')':
                layer = layer + ")"

            if (layer[0:layer.find("(")] == 'DENSE' or layer[0:layer.find(
                    "(")] == 'SOFTMAX') and should_flatten and layer_num != 1:
                model.add(Flatten())
                print("added flattened")
                should_flatten = False

            model.add(self.get_layer(layer, input_dim, layer_num))

            if (layer[0:layer.find("(")] == 'DENSE' or layer[0:layer.find("(")] == 'SOFTMAX') and layer_num <= len(
                    layers) - 1:
                model.add(Dropout(0.2))
                print("added drouput " + str(layer))

            layer_num = layer_num + 1

        model.summary()

        if AppContext.GPUS_TO_USE >= 2:
            parallel_model = multi_gpu_model(model, gpus=AppContext.GPUS_TO_USE)
            parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return parallel_model
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
