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


@app.route('/model-train-epoch-update', methods=['POST'])
def update_model_training_epoch_status():
    data = json.loads(request.data)
    print(data)
    StatusService().log_epoch_update(data)

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

    client_id = StatusService().register_client(data['host'], data['port'])
    data['status'] = 'registered'
    data['client_id'] = client_id

    return app.response_class(
        response=json.dumps(data),
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


def set_context(args):
    AppContext.SERVER_PORT = int(args.server_port)
    AppContext.DATASET_DIR = args.data_dir
    AppContext.METADATA_DIR = args.metadata_dir
    AppContext.REPLAY_FROM_METASTORE = bool(args.replay_from_metastore)

    if not AppContext.REPLAY_FROM_METASTORE:
        print('REPLAY_FROM_METASTORE option not enabled, deleteing metastore directory')
        shutil.rmtree(AppContext.METADATA_DIR)

    if not os.path.exists(AppContext.METADATA_DIR):
        print("meta directory doesn't exist, creating one!")
        os.mkdir(AppContext.METADATA_DIR)

    AppContext.EPOCH_STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'epoch_status.txt')
    AppContext.MODELS_FOLDER = os.path.join(AppContext.METADATA_DIR, 'models')
    AppContext.CLIENTS_FOLDER = os.path.join(AppContext.METADATA_DIR, 'clients')
    AppContext.SERVER_STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'server_status.txt')

    AppContext.CURRENT_MAX_CLIENT_ID = 0
    AppContext.CURRENT_MAX_ITERATION = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-server_port', help='The port on which Server can be contacted')
    parser.add_argument('-data_dir', help='The directory on the server where state space params exists')
    parser.add_argument('-metadata_dir', help='The directory on the client machine will be used as metadata repo')
    parser.add_argument('-replay_from_metastore', help='Option to clear metastore or update current state from it')

    set_context(parser.parse_args())

    app.run(host="0.0.0.0", port=AppContext.SERVER_PORT)
