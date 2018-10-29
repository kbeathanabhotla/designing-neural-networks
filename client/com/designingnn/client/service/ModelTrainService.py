from com.designingnn.client.service.StatusUpdateService import StatusUpdateService


class ModelTrainService:
    def __init__(self):
        self.status_update_service = StatusUpdateService()

    def train_model(self, model_options):
        model_def = model_options['model_def']
        print("model getting trained : {} ".format(model_def))

        self.set_client_status('training')

        self.set_client_status('free')

        pass

    def set_client_status(self, status):
        if status == 'training':
            if self.status_update_service.get_client_status() == 'free':
                self.status_update_service.update_client_status(status)
            else:
                raise ValueError('Cannot set status for busy client')
        elif status == 'free':
            self.status_update_service.update_client_status(status)
