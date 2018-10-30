import argparse
import socket
import os
import requests
from flask import Flask, json, request
from requests.exceptions import ConnectionError
import shutil

from com.designingnn.client.core import AppContext
from com.designingnn.client.service.ModelTrainService import ModelTrainService
from com.designingnn.client.service.StatusService import StatusService

app = Flask(__name__)


@app.route('/')
def test_endpoint():
    return "This is a test endpoint!!"


@app.route('/status')
def get_status():
    data = {
        'status': StatusService().get_client_status()
    }

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )


@app.route('/train-model', methods=['POST'])
def train_model():
    model_options = json.loads(request.data)

    print("received model options for training {}".format(model_options))
    ModelTrainService().train_model(model_options)

    return app.response_class(
        response=json.dumps(model_options),
        status=201,
        mimetype='application/json'
    )


@app.route('/get-trained-model-list')
def get_status():
    return app.response_class(
        response=json.dumps(StatusService().get_trained_models_list()),
        status=200,
        mimetype='application/json'
    )


@app.route('/get-model-train-status/<model_id>', methods=['GET'])
def get_model_training_status(model_id):
    return app.response_class(
        response=json.dumps(StatusService().get_model_train_status(model_id)),
        status=200,
        mimetype='application/json'
    )


def ping_server(host, port):
    try:
        r = requests.get(url="http://{}:{}".format(host, port))
        return r.status_code
    except ConnectionError:
        return 502


def register_client(server_host, server_port, client_port):
    data = {
        'host': socket.gethostname(),
        'port': int(client_port)
    }

    try:
        r = requests.post(url="http://{}:{}/register".format(server_host, server_port), data=data)
        return r.status_code
    except ConnectionError:
        return 502


def set_context_for_current_client():
    if os.path.exists(AppContext.METADATA_DIR):
        print("client meta directory already exists, clearing it!")
        shutil.rmtree(AppContext.METADATA_DIR)

    print("meta directory for client created!")
    os.mkdir(AppContext.METADATA_DIR)

    AppContext.STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'status.txt')
    AppContext.MODELS_INFO_FOLDER = os.path.join(AppContext.METADATA_DIR, 'trained_models_info')

    print("status file created.")
    with open(AppContext.STATUS_FILE, 'w+') as status_f:
        status_f.write('{}'.format('free'))

    print("trained_models_info_folder created.")
    os.mkdir(AppContext.MODELS_INFO_FOLDER)


def set_context_options(args):
    AppContext.SERVER_HOST = args.server_host
    AppContext.SERVER_PORT = args.server_port
    AppContext.CLIENT_PORT = int(args.server_port)
    AppContext.DATASET_DIR = args.data_dir
    AppContext.METADATA_DIR = args.metadata_dir
    AppContext.GPUS_TO_USE = args.num_gpus_to_use


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-server_host', help='The hostname/IP on which Server is reachable')
    parser.add_argument('-client_port', help='The port on which the current client has to run')
    parser.add_argument('-data_dir', help='The directory on the client machine on which dataset exists')
    parser.add_argument('-metadata_dir', help='The directory on the client machine will be used as metadata repo')
    parser.add_argument('-num_gpus_to_use', help='Number of GPUs that would be used by the client for training models')

    set_context_options(parser.parse_args())

    server_ping_status_code = ping_server(AppContext.SERVER_HOST, AppContext.SERVER_PORT)

    if server_ping_status_code == 200:
        print("server ping successful.")
        register_client(AppContext.SERVER_HOST, AppContext.SERVER_PORT, AppContext.CLIENT_PORT)
        print("registered client to server.")

        set_context_for_current_client()

        print("starting client on port {}".format(AppContext.CLIENT_PORT))
        app.run(host="0.0.0.0", port=AppContext.CLIENT_PORT)
    else:
        print("Server ping failed. status code returned {}".format(server_ping_status_code))
