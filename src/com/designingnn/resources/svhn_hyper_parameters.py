import svhn_state_space_parameters as ssp

MODEL_NAME = 'svhn'

# Number of output neurons
NUM_CLASSES = 10                                                                    # Number of output neurons

#Final Image Size
IMAGE_HEIGHT = 32                                                                   # Final Image Height NOTE: code only supports square images right now o_O
IMAGE_WIDTH = 32                                                                    # Final Image Width  NOTE: code only supports square images right now o_O

#Batch Queue parameters
TRAIN_BATCH_SIZE = 128                                                              # Batch size for training
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 68257                                            # Number of training examples
NUM_ITER_PER_EPOCH_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAIN_BATCH_SIZE      
EVAL_BATCH_SIZE = 100                                                               # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000                                              # Number of validation examples
NUM_ITER_TO_TRY_LR = NUM_ITER_PER_EPOCH_TRAIN                                       # Number of iterations to try learning rate (based on ACC_THRESHOLD)
                                                                                    # Please make an integer multiple of NUM_ITER_PER_EPOCH_TRAIN

TEST_INTERVAL_EPOCHS = 1                                                            # Num epochs to test on, should really always be 1                                      
MAX_EPOCHS = 20                                                                     # Max number of epochs to train model
MAX_STEPS = MAX_EPOCHS * NUM_ITER_PER_EPOCH_TRAIN                                   


#Training Parameters
OPTIMIZER = 'Adam'                                                                  # Optimizer (should be in caffe format string)
MOMENTUM = 0.9                                                                      # Momentum
WEIGHT_DECAY_RATE = 0.0005                                                          # Weight decay factor   


# Learning Rate
INITIAL_LEARNING_RATES = [0.001 * (0.4**i) for i in range(5)]                       # List of initial learning rates to try before giving up on model
ACC_THRESHOLD = 0.15                                                                # Model must achieve greater than ACC_THRESHOLD performance in NUM_ITER_TO_TRY_LR or it is killed and next initial learning rate is tried.
LEARNING_RATE_DECAY_FACTOR = 0.2                                                    # Learning rate decay factor.
NUM_EPOCHS_PER_DECAY = 5                                                            # Epochs after which learning rate decays.
LR_POLICY = "step"                                                                  # Caffe Default

SATISFIED_ACCURACY = 0.92
