MODEL_NAME = 'mnist'

# Number of output neurons
NUM_CLASSES = 10

# Final Image Size
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

# Batch Queue parameters
TRAIN_BATCH_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 55000
NUM_ITER_PER_EPOCH_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000
NUM_ITER_TO_TRY_LR = NUM_ITER_PER_EPOCH_TRAIN

TEST_INTERVAL_EPOCHS = 1
MAX_EPOCHS = 10
MAX_STEPS = MAX_EPOCHS * NUM_ITER_PER_EPOCH_TRAIN

# Training Parameters
OPTIMIZER = 'Adam'
MOMENTUM = 0.9
WEIGHT_DECAY_RATE = 0.0005

# Learning Rate
INITIAL_LEARNING_RATES = [0.001 * (0.4 ** i) for i in range(5)]
ACC_THRESHOLD = 0.15
LEARNING_RATE_DECAY_FACTOR = 0.2
NUM_EPOCHS_PER_DECAY = 5
LR_POLICY = "step"
