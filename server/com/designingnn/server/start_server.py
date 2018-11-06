import argparse
import json
import sys
import os
import shutil

from flask import Flask, request

from com.designingnn.server.core import AppContext
from com.designingnn.server.service.StatusService import StatusService

app = Flask(__name__)


@app.route('/')
def test_endpoint():
    return "This is a test endpoint!!"


@app.route('/model-train-epoc-update', methods=['POST'])
def update_model_training_epoc_status():
    data = json.loads(request.data)
    print(data)
    StatusService().update_epoc_status(data)

    return app.response_class(
        response=json.dumps(data),
        status=201,
        mimetype='application/json'
    )


@app.route('/model-train-status', methods=['POST'])
def update_model_training_status():
    data = json.loads(request.data)
    print(data)
    StatusService().update_model_training_status(data)

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )


@app.route('/register', methods=['POST'])
def register_client():
    data = json.loads(request.data)
    print(data)

    StatusService().add_client(data['host'], data['port'])
    data['status'] = 'registered'

    return app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )


@app.route('/get-clients', methods=['GET'])
def register_client():
    return app.response_class(
        response=json.dumps(StatusService().get_clients()),
        status=200,
        mimetype='application/json'
    )


@app.route('/stats', methods=['GET'])
def get_stats():
    return app.response_class(
        response=json.dumps(StatusService().get_server_stats()),
        status=200,
        mimetype='application/json'
    )


def set_context_for_current_client():

    if not os.path.exists(AppContext.METADATA_DIR):
        print("meta directory doesn't exist, creating one!")
        os.mkdir(AppContext.METADATA_DIR)

    AppContext.CLIENTS_FILE = os.path.join(AppContext.METADATA_DIR, 'clients.txt')
    AppContext.EPOC_STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'epoc_status.txt')
    AppContext.SERVER_STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'server_status.txt')

    # AppContext.MODELS_INFO_FOLDER = os.path.join(AppContext.METADATA_DIR, 'trained_models_info')
    #
    # print("status file created.")
    # with open(AppContext.STATUS_FILE, 'w+') as status_f:
    #     status_f.write('{}'.format('free'))
    #
    # print("trained_models_info_folder created.")
    # os.mkdir(AppContext.MODELS_INFO_FOLDER)


def set_context_options(args):
    AppContext.SERVER_PORT = int(args.server_port)
    AppContext.DATASET_DIR = args.data_dir
    AppContext.METADATA_DIR = args.metadata_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-data_dir', help='The directory on the server where state space params exists')
    parser.add_argument('-metadata_dir', help='The directory on the client machine will be used as metadata repo')

    set_context_options(parser.parse_args())

    app.run(host="0.0.0.0", port=int(sys.argv[1]))
