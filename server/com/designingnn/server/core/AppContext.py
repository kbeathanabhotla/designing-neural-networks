import socket

DATASET_DIR = None
METADATA_DIR = None
SERVER_PORT = None
# CLIENTS_FILE = None
EPOCH_STATUS_FILE = None
MODELS_FOLDER = None
# SERVER_STATUS_FILE = None
#
# CLIENTS = []
#
# CURRENT_EPSILON = None
# CURRENT_ITERATION = None
#
MODELS_IN_TRAINING = []


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))

IP_ADDRESS = s.getsockname()[0]
