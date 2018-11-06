import json

from com.designingnn.server.core import AppContext


class StatusService:
    def __init__(self):
        pass

    def update_model_training_status(self, status):
        pass

    def update_server_status(self):
        status = {
            'model_under_training': [],
            'current_epsilon': AppContext.CURRENT_EPSILON
        }
        with open(AppContext.SERVER_STATUS_FILE, 'a+') as server_status_file:
            server_status_file.write(json.dumps(status))

    def update_epoc_status(self, status):
        with open(AppContext.EPOC_STATUS_FILE, 'a+') as epoc_status_file:
            epoc_status_file.write(json.dumps(status))

    def get_server_stats(self):
        dataset_path = AppContext.DATASET_DIR
        clients = self.get_clients()

    def get_clients(self):
        return AppContext.CLIENTS

    def add_client(self, host, port):
        client = {
            'host': host,
            'port': port,
            'status': 'free'
        }

        AppContext.CLIENTS.append(client)

        with open(AppContext.CLIENTS_FILE, 'a+') as clients_file:
            clients_file.write(json.dumps({
                'host': host,
                'port': port,
                'status': 'free'
            }))
