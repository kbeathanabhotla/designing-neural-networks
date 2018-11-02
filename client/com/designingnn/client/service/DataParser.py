from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.utils import multi_gpu_model


def get_layer(layer, layer_num = None, inut_dim):
    layer_name = layer[0:layer.find("(")]
    print layer_name

    if layer_name.equals('SOFTMAX'):
        num_out_neurons = int(layer_name.replace('SOFTMAX', '').replace('(', '').replace(')', '').strip())
        return Dense(num_out_neurons, kernel_initializer='normal', activation='softmax')

    elif layer_name.equals('CONV'):
        params = layer_name.replace('CONV', '').replace('(', '').replace(')', '').strip().split(',')
        num_filters = int(params[0])
        filter_size = int(params[1])
        stride = int(params[2])

        return Conv2D(num_filters, (filter_size, filter_size), strides=(stride, stride), padding='same',
                      activation='relu')

    elif layer_name.equals('MAXPOOLING'):
        params = layer_name.replace('MAXPOOLING', '').replace('(', '').replace(')', '').strip().split(',')
        pool_size = int(params[0])
        stride = int(params[1])

        return MaxPooling2D((pool_size, pool_size), strides=(stride, stride))

    elif layer_name.equals('DENSE') and :
        params = layer_name.replace('DENSE', '').replace('(', '').replace(')', '').strip().split(',')
        num_output = int(params[0])

        return Dense(num_output, activation='relu')


def generate_model(model_def):
    model_def = model_def[1:-1]
    print model_def

    model = Sequential()

    layers = model_def.split("),")
    for layer in layers:
        layer = layer.strip()
        if layer[-1] != ')':
            layer = layer + ")"

        get_layer(layer)

        #     model.adDENSE(get_layer(layer))
        #
        # model.summary()
        #
        # parallel_model = multi_gpu_model(model, gpus=8)
        # parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # return parallel_model


if __name__ == '__main__':
    str1 = '[CONV(256,5,1), CONV(256,3,1), DENSE(1,4), CONV(64,5,1), CONV(256,1,1), DENSE(2,4), CONV(512,3,1), CONV(128,3,1), DENSE(3,4), CONV(128,1,1), CONV(512,3,1), DENSE(4,4), CONV(256,3,1), MAXPOOLING(10), SOFTMAX(10)]'
    str2 = '[CONV(256,1,1), CONV(64,3,1), DENSE(1,2), CONV(512,5,1), CONV(128,1,1), DENSE(2,2), CONV(256,5,1), MAXPOOLING(10), SOFTMAX(10)]'
    str3 = '[CONV(128,3,1), CONV(64,1,1), DENSE(1,2), CONV(64,5,1), CONV(64,3,1), DENSE(2,2), MAXPOOLING(10), SOFTMAX(10)]'

    # generate_model(str1)
    # generate_model(str2)
    generate_model(str3)
