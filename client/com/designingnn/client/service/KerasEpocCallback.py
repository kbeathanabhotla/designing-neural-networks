import json

import requests
from keras.callbacks import Callback

from com.designingnn.client.core import AppContext


class KerasEpochCallback(Callback):
    def __init__(self, model_id, test_data):
        super(KerasEpochCallback, self).__init__()
        self.test_data = test_data
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)

        print('\nTesting data -> epoch: {}, test_loss: {}, test_acc: {}\n'.format(epoch, loss, acc))

        data = {
            'epoch': epoch,
            'test_accuracy': acc,
            'model_id': self.model_id,
            'client_hostname': AppContext.IP_ADDRESS
        }

        requests.post("http://{}:{}/model-train-epoch-update".format(AppContext.SERVER_HOST, AppContext.SERVER_PORT),
                      json.dumps(data))
