import os
from pandas import json

from com.designingnn.client.core import AppContext


class StatusService:
    def __init__(self):
        pass

    def get_client_status(self):
        status = None
        with open(AppContext.STATUS_FILE, 'r') as status_f:
            status = status_f.read()

        return status.strip()

    def update_client_status(self, status):
        with open(AppContext.STATUS_FILE, 'w+') as status_f:
            status_f.write('{}'.format(status))

    def get_trained_models_list(self):
        models = []
        for model_info_file in os.listdir(AppContext.MODELS_INFO_FOLDER):
            with open(os.path.join(AppContext.MODELS_INFO_FOLDER, model_info_file), 'r') as trained_model_info:
                model_info = trained_model_info.read()
                models.append(json.loads(model_info))

        return models

    def update_model_training_info(self, model_info):
        model_info_file_path = os.path.join(AppContext.MODELS_INFO_FOLDER, '{}.txt'.format(model_info['model_id']))
        with open(model_info_file_path, 'w+') as model_info_file:
            model_info_file.write(json.dumps(model_info))

    def get_model_train_status(self, model_id):
        status = None
        model_info_file_path = os.path.join(AppContext.MODELS_INFO_FOLDER, '{}.txt'.format(model_id))
        with open(model_info_file_path, 'r') as model_info_file:
            status = model_info_file.read()

        return status


if __name__ == '__main__':
    AppContext.METADATA_DIR = '/mnt/D/Learning/MTSS/Sem4/code/designing-neural-networks/meta_repo/client1'
    AppContext.MODELS_INFO_FOLDER = '/mnt/D/Learning/MTSS/Sem4/code/designing-neural-networks/meta_repo/client1/trained_models_info'
    AppContext.STATUS_FILE = '/mnt/D/Learning/MTSS/Sem4/code/designing-neural-networks/meta_repo/client1/status.txt'

    import shutil

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

    status_service = StatusService()

    # print status_service.get_client_status()
    # StatusService().update_client_status('busy')
    # print StatusService().get_client_status()

    l = status_service.get_trained_models_list()
    print(len(l))
    for itm in l:
        print itm
