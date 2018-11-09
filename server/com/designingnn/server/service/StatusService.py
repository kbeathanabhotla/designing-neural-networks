import json
import os

from com.designingnn.server.core import AppContext
from com.designingnn.server.service.QLearnerService import QLearnerService
import requests

class StatusService:
    def __init__(self):
        pass

    # def log_epoch_update(self, update):
    #     for idx, model in enumerate(AppContext.MODELS_IN_TRAINING):
    #         if model['id'] == update['model_id']:
    #             model['epoch'] = update['epoch']
    #
    #     with open(AppContext.EPOCH_STATUS_FILE, 'a+') as epoch_status_file:
    #         epoch_status_file.write(json.dumps(update) + '\n')
    #
    def log_model_training_update(self, update):
        if 'stack_trace' in update:
            update['stack_trace'] = update['stack_trace'].encode('base64').strip()

        if 'status' in update and (update['status'] == 'failed' or update['status'] == 'completed'):
            AppContext.MODELS_IN_TRAINING = [model for model in AppContext.MODELS_IN_TRAINING if
                                             model['id'] != update['model_id']]

        f = os.path.join(AppContext.MODELS_FOLDER, '{}.txt'.format(update['model_id']))
        with open(f, 'w+') as model_file:
            model_file.write(json.dumps(update) + '\n')

        self.train_new_model()

    def register_client(self, host, port):
        AppContext.CURRENT_MAX_CLIENT_ID = AppContext.CURRENT_MAX_CLIENT_ID + 1
        client_info = {
            'host': host,
            'port': port,
            'status': 'free',
            'id': AppContext.CURRENT_MAX_CLIENT_ID
        }

        AppContext.CLIENTS.append(client_info)

        self.update_client_status(client_info, 'free')
        self.train_new_model()

    def update_client_info_on_disk(self, client_info):
        with open(os.path.join(AppContext.CLIENTS_FOLDER, '{}.txt'.format(client_info['id'])), 'w+') as clients_file:
            clients_file.write(json.dumps(client_info))

    def update_client_status(self, client_id, status):
        for idx, client in enumerate(AppContext.CLIENTS):
            if client['id'] == client_id:
                client['status'] = status

            self.update_client_info_on_disk(client)

    def get_clients_info_from_disk(self):
        clients = []
        for client_info_file in os.listdir(AppContext.CLIENTS_FOLDER):
            with open(os.path.join(AppContext.CLIENTS_FOLDER, client_info_file), 'r') as client_info:
                clients.append(json.loads(client_info.read()))

        AppContext.CLIENTS = {client['id']: client for client in clients}

    def get_server_stats(self):
        return {
            'dataset_path': AppContext.DATASET_DIR,
            'metadata_path': AppContext.METADATA_DIR,
            'clients': [AppContext.CLIENTS[client_id] for client_id in AppContext.CLIENTS],
            'model_under_training': [AppContext.MODELS_IN_TRAINING[model_id] for model_id in
                                     AppContext.MODELS_IN_TRAINING],
            'current_max_iter': AppContext.CURRENT_MAX_ITERATION
        }

    def update_model_status(self, model_id, status):
        for idx, model in enumerate(AppContext.MODELS_IN_TRAINING):
            if model['model_id'] == model_id:
                model['status'] = status

            self.update_model_info_on_disk(model)

    def update_model_info_on_disk(self, model_info):
        with open(os.path.join(AppContext.MODELS_FOLDER, '{}.txt'.format(model_info['id'])), 'w+') as model_file:
            model_file.write(json.dumps(model_info))

    def train_new_model(self):
        for idx, client in enumerate(AppContext.CLIENTS):
            if client['status'] == 'free':
                q_learner_service = QLearnerService()

                model_def = q_learner_service.generate_new_model()
                model_id = AppContext.CURRENT_MAX_ITERATION

                if model_def in AppContext.MODELS_GENERATED:
                    pass

                AppContext.MODELS_GENERATED[model_def] = model_id
                AppContext.MODELS_IN_TRAINING[model_id] = {
                    'client_id': client['id'],
                    'model_id': model_id,
                    'model_def': model_def
                }
                AppContext.CURRENT_MAX_ITERATION = AppContext.CURRENT_MAX_ITERATION + 1
                self.update_client_status(client['id'], 'training')

                self.update_model_info_on_disk(AppContext.MODELS_IN_TRAINING[model_id])

                requests.post("http://{}:{}/train-model", data=AppContext.MODELS_IN_TRAINING[model_id])
            else:
                print('None of the clients are free to train new model!')
