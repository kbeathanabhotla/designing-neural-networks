from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.utils import multi_gpu_model

from com.designingnn.client.core import AppContext


class ModelParser:
    def __init__(self, model_def, input_dim):
        self.model_def = model_def
        self.input_dim = input_dim
        self.layers = self.parse_model_def()
        self.first_layer_type = self.layers[0]['layer_name']
        self.first_dense_layer_num = self.get_first_dense_layer_num()
        self.last_layer_num = self.layers[-1]['layer_num']

        print('Training {}'.format(self.model_def))

    def parse_model_def(self):
        parsed_layers = []

        layers = self.model_def[1:-1].split("),")
        layers = map(lambda layer: layer.strip(), layers)
        layers = map(lambda layer: layer if layer[-1] == ')' else layer + ")", layers)

        for idx, layer_info in enumerate(layers):
            layer_def = {}

            layer_name = layer_info[0:layer_info.find("(")]
            layer_def['layer_name'] = layer_name

            layer_params = layer_info[layer_info.find('(') + 1: layer_info.find(')')]

            if layer_name == 'DENSE' or layer_name == 'SOFTMAX':
                layer_def['num_neurons'] = int(layer_params)

            elif layer_name == 'CONV':
                params = layer_params.strip().split(',')
                layer_def['num_filters'] = int(params[0])
                layer_def['filter_size'] = int(params[1])
                layer_def['stride'] = int(params[2])

            elif layer_name == 'AVGPOOLING' or layer_name == 'MAXPOOLING':
                params = layer_params.strip()
                layer_def['pool_size'] = int(params)

            layer_def['layer_num'] = idx

            parsed_layers.append(layer_def)

        # for parsed_layer in parsed_layers:
        #     print parsed_layer

        return parsed_layers

    def get_first_dense_layer_num(self):
        for layer in self.layers:
            if layer['layer_name'] == 'DENSE' or layer['layer_name'] == 'SOFTMAX':
                return layer['layer_num']
        return -1

    def is_first_layer_dense(self):
        return self.first_layer_type == 'DENSE' or self.first_layer_type == 'SOFTMAX'

    def get_layer(self, layer, input_dim=None):
        layer_name = layer['layer_name']

        if layer_name == 'SOFTMAX':
            num_out_neurons = layer['num_neurons']
            if layer['layer_num'] != 0:
                return Dense(num_out_neurons, kernel_initializer='normal', activation='softmax')
            elif self.is_first_layer_dense():
                return Dense(num_out_neurons, input_dim=input_dim,
                             kernel_initializer='normal', activation='softmax')

        elif layer_name == 'CONV':
            num_filters = layer['num_filters']
            filter_size = layer['filter_size']
            stride = layer['stride']

            if layer['layer_num'] != 0:
                return Conv2D(num_filters, (filter_size, filter_size), strides=(stride, stride), padding='same',
                              activation='relu')
            else:
                return Conv2D(num_filters, (filter_size, filter_size), input_shape=input_dim, strides=(stride, stride),
                              padding='same',
                              activation='relu')

        elif layer_name == 'MAXPOOLING':
            pool_size = layer['pool_size']
            return MaxPooling2D((pool_size, pool_size))

        elif layer_name == 'AVGPOOLING':
            pool_size = layer['pool_size']
            return AveragePooling2D((pool_size, pool_size))

        elif layer_name == 'DENSE':
            num_output = layer['num_neurons']

            if layer['layer_num'] != 0:
                return Dense(num_output, activation='relu')
            elif self.is_first_layer_dense():
                return Dense(num_output, input_dim=input_dim, activation='relu')

    def generate_model(self, input_dim):
        model = Sequential()

        for layer in self.layers:
            if self.first_dense_layer_num == layer['layer_num'] and not self.is_first_layer_dense():
                print("added flattened")
                model.add(Flatten())

            model.add(self.get_layer(layer, input_dim))

            if (layer['layer_name'] == 'DENSE' or layer['layer_name'] == 'SOFTMAX') and layer[
                'layer_num'] < self.last_layer_num:
                model.add(Dropout(0.2))
                print("added dropout after " + str(layer))

        model.summary()

        if AppContext.GPUS_TO_USE >= 2:
            parallel_model = multi_gpu_model(model, gpus=AppContext.GPUS_TO_USE)
            parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return parallel_model
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
