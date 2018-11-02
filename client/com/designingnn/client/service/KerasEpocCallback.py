from keras.callbacks import Callback
import requests

from com.designingnn.client.core import AppContext


class KerasEpocCallback(Callback):
    def __init__(self, model_id, test_data):
        self.test_data = test_data
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)

        print('\nTesting epoc: {}, loss: {}, acc: {}\n'.format(self.params['nb_epoch'], loss, acc))

        data = {
            'epoc': self.params['nb_epoch'],
            'test_accuracy': acc,
            'model_id': self.model_id,
            'client_hostname': AppContext.IP_ADDRESS
        }

        requests.post("http://{}:{}/model-train-epoc-update", data)
