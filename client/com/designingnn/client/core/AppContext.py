import socket

DATASET_DIR = None
METADATA_DIR = None
SERVER_PORT = None
SERVER_HOST = None
CLIENT_PORT = None
GPUS_TO_USE = 1

STATUS_FILE = None
MODELS_INFO_FOLDER = None

HOSTNAME = socket.gethostname()
IP_ADDRESS = socket.gethostbyname(socket.gethostname())
