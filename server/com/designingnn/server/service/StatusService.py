import json
import os

from com.designingnn.server.core import AppContext
from com.designingnn.server.service.QLearnerService import QLearnerService


class StatusService:
    def __init__(self):
        pass

    def log_epoch_update(self, update):
        for idx, model in enumerate(AppContext.MODELS_IN_TRAINING):
            if model['id'] == update['model_id']:
                model['epoch'] = update['epoch']

        with open(AppContext.EPOCH_STATUS_FILE, 'a+') as epoch_status_file:
            epoch_status_file.write(json.dumps(update))

    def log_model_training_update(self, update):
        file = os.path.join(AppContext.MODELS_FOLDER, '{}.txt'.format(update['model_id']))
        with open(file, 'w+') as model_file:
            model_file.write(json.dumps(update))



    # def update_model_training_status(self, status):
    #     AppContext.MODELS_IN_TRAINING = [x for x in AppContext.MODELS_IN_TRAINING if
    #                                      x['model_id'] != status['model_id']]
    #
    #
    #
    # def update_server_status(self):
    #     status = {
    #         'model_under_training': AppContext.MODELS_IN_TRAINING,
    #         'current_epsilon': AppContext.CURRENT_EPSILON,
    #         'current_iter': AppContext.CURRENT_ITERATION
    #     }
    #     with open(AppContext.SERVER_STATUS_FILE, 'a+') as server_status_file:
    #         server_status_file.write(json.dumps(status))
    #
    # def update_models_in_training(self, model_def, model_id, client_host):
    #     AppContext.MODELS_IN_TRAINING.append({
    #         'model_def': model_def,
    #         'model_id': model_id,
    #         'host': client_host
    #     })
    #
    # def update_epoc_status(self, status):
    #
    #     with open(AppContext.EPOC_STATUS_FILE, 'a+') as epoc_status_file:
    #         epoc_status_file.write(json.dumps(status))
    #
    # def get_server_stats(self):
    #     return {
    #         'dataset_path': AppContext.DATASET_DIR,
    #         'clients': self.get_clients(),
    #         'current_iter': AppContext.CURRENT_ITERATION,
    #         'model_under_training': AppContext.MODELS_IN_TRAINING
    #     }
    #
    # def get_clients(self):
    #     return AppContext.CLIENTS
    #
    # def add_client(self, host, port):
    #     client = {
    #         'host': host,
    #         'port': port,
    #         'status': 'free'
    #     }
    #
    #     AppContext.CLIENTS.append(client)
    #
    #     with open(AppContext.CLIENTS_FILE, 'a+') as clients_file:
    #         clients_file.write(json.dumps({
    #             'host': host,
    #             'port': port,
    #             'status': 'free'
    #         }))
    #
    # def update_client_status(self, host, status):
    #     clients = []
    #     with open(AppContext.CLIENTS_FILE, 'r') as clients_file:
    #         lines = clients_file.readlines()
    #
    #         for line in lines:
    #             clients.append(json.loads(line))
    #
    #     for client in clients:
    #         if client['host'] == host:
    #             client['status'] = status
    #
    #     with open(AppContext.CLIENTS_FILE, 'w+') as clients_file:
    #         clients_file.write(json.dumps(clients))
    #
    # def train_new_model(self):
    #     for client in AppContext.CLIENTS:
    #         if client['status'] == 'free':
    #             AppContext.CURRENT_ITERATION = AppContext.CURRENT_ITERATION + 1
    #
    #             self.update_client_status(client['host'], 'training')
    #             client['status'] = 'training'
    #
    #             model_def = QLearnerService().generate_new_model()
    #             model_id = AppContext.CURRENT_ITERATION
