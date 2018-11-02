class ModelParser:
    def __init__(self):
        pass

    def parse_model(self, model_def):
        net_list = parse('net', model_def)
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
                print self.hyper_parameters.NUM_CLASSES

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

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

