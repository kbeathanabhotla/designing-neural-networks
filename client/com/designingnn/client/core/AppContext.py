import socket

DATASET_DIR = None
METADATA_DIR = None
SERVER_PORT = None
SERVER_HOST = None
CLIENT_PORT = None
GPUS_TO_USE = 1

STATUS_FILE = None
MODELS_INFO_FOLDER = None

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))

HOSTNAME = socket.gethostname()
IP_ADDRESS = s.getsockname()[0]
