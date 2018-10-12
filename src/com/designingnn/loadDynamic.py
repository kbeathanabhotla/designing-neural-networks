import importlib

from com.designingnn.resources import mnist_state_space_parameters

fl = '/mnt/D/Learning/MTSS/Sem 4/code/designing-neural-networks/src/com/designingnn/resources/mnist_state_space_parameters.py'

if 'mnist':
    state_space_params = mnist_state_space_parameters


print state_space_params.allow_consecutive_pooling
