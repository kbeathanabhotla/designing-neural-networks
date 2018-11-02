# from keras.layers.convolutional import Conv2D
# from keras.layers.core import Dense, Dropout
# from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras.models import Sequential
# from keras.utils import multi_gpu_model
#
# from com.designingnn.client.core import AppContext
#
#
# def get_layer(layer, input_dim=None, layer_num=None):
#     layer_name = layer[0:layer.find("(")]
#     print layer_name
#
#     if layer_name == 'SOFTMAX':
#         num_out_neurons = int(layer.replace('SOFTMAX', '').replace('(', '').replace(')', '').strip())
#
#         if layer_num != 1:
#             return Dense(num_out_neurons, kernel_initializer='normal', activation='softmax')
#         else:
#             return Dense(num_out_neurons, input_shape=input_dim, kernel_initializer='normal', activation='softmax')
#
#     elif layer_name == 'CONV':
#         params = layer.replace('CONV', '').replace('(', '').replace(')', '').strip().split(',')
#         num_filters = int(params[0])
#         filter_size = int(params[1])
#         stride = int(params[2])
#
#         if layer_num != 1:
#             return Conv2D(num_filters, (filter_size, filter_size), strides=(stride, stride), padding='same',
#                           activation='relu')
#         else:
#             return Conv2D(num_filters, (filter_size, filter_size), input_shape=input_dim, strides=(stride, stride),
#                           padding='same',
#                           activation='relu')
#
#     elif layer_name == 'MAXPOOLING':
#         pool_size = int(layer.replace('MAXPOOLING', '').replace('(', '').replace(')', '').strip())
#         return MaxPooling2D((pool_size, pool_size))
#
#     elif layer_name == 'AVGPOOLING':
#         pool_size = int(layer.replace('AVGPOOLING', '').replace('(', '').replace(')', '').strip())
#         return AveragePooling2D((pool_size, pool_size))
#
#     elif layer_name == 'DENSE':
#         params = layer.replace('DENSE', '').replace('(', '').replace(')', '').strip().split(',')
#         num_output = int(params[0])
#
#         if layer_num != 1:
#             return Dense(num_output, activation='relu')
#         else:
#             return Dense(num_output, input_shape=input_dim, activation='relu')
#
#
# def generate_model(model_def, input_dim):
#     model_def = model_def[1:-1]
#     print model_def
#
#     model = Sequential()
#
#     layers = model_def.split("),")
#     layer_num = 1
#     for layer in layers:
#         layer = layer.strip()
#         if layer[-1] != ')':
#             layer = layer + ")"
#
#         model.add(get_layer(layer, input_dim, layer_num))
#
#         if layer[0:layer.find("(")] == 'DENSE':
#             model.add(Dropout(0.2))
#
#         layer_num = layer_num + 1
#
#     model.summary()
#
#     if AppContext.GPUS_TO_USE >= 2:
#         parallel_model = multi_gpu_model(model, gpus=AppContext.GPUS_TO_USE)
#         parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return parallel_model
#     else:
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model
#
#
# if __name__ == '__main__':
#     str1 = '[CONV(256,5,1), CONV(256,3,1), DENSE(1,4), CONV(64,5,1), CONV(256,1,1), DENSE(2,4), CONV(512,3,1), CONV(128,3,1), DENSE(3,4), CONV(128,1,1), CONV(512,3,1), DENSE(4,4), CONV(256,3,1), MAXPOOLING(10), SOFTMAX(10)]'
#     str2 = '[CONV(256,1,1), CONV(64,3,1), DENSE(1,2), CONV(512,5,1), CONV(128,1,1), DENSE(2,2), CONV(256,5,1), MAXPOOLING(10), SOFTMAX(10)]'
#     str3 = '[CONV(128,3,1), CONV(64,1,1), DENSE(1,2), CONV(64,5,1), CONV(64,3,1), DENSE(2,2), MAXPOOLING(10), SOFTMAX(10)]'
#
#     # generate_model(str1)
#     # generate_model(str2)
#     generate_model(str3, (28, 28, 1))
