class State:
    def __init__(self,
                 layer_type=None,  # String -- conv, pool, fc, softmax
                 layer_depth=None,  # Current depth of network
                 filter_depth=None,  # Used for conv, 0 when not conv
                 filter_size=None,  # Used for conv and pool, 0 otherwise
                 stride=None,  # Used for conv and pool, 0 otherwise
                 image_size=None,  # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 fc_size=None,  # Used for fc and softmax -- number of neurons in layer
                 terminate=None,
                 state_list=None):  # can be constructed from a list instead, list takes precedent
        if not state_list:
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.filter_depth = filter_depth
            self.filter_size = filter_size
            self.stride = stride
            self.image_size = image_size
            self.fc_size = fc_size
            self.terminate = terminate
        else:
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.filter_depth = state_list[2]
            self.filter_size = state_list[3]
            self.stride = state_list[4]
            self.image_size = state_list[5]
            self.fc_size = state_list[6]
            self.terminate = state_list[7]

    def as_tuple(self):
        return (self.layer_type,
                self.layer_depth,
                self.filter_depth,
                self.filter_size,
                self.stride,
                self.image_size,
                self.fc_size,
                self.terminate)

    def as_list(self):
        return list(self.as_tuple())

    def copy(self):
        return State(self.layer_type,
                     self.layer_depth,
                     self.filter_depth,
                     self.filter_size,
                     self.stride,
                     self.image_size,
                     self.fc_size,
                     self.terminate)
