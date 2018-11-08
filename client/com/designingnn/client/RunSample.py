from com.designingnn.client.core import AppContext
from com.designingnn.client.service.ModelParseAndTrainTask import ModelParseAndTrainTask
import os

AppContext.METADATA_DIR = "/mnt/D/Learning/MTSS/Sem4/code/designing-neural-networks/meta_repo/client_1"
AppContext.DATASET_DIR = "/mnt/D/Learning/MTSS/Sem4/code/designing-neural-networks/meta_repo/mnist"

AppContext.STATUS_FILE = os.path.join(AppContext.METADATA_DIR, 'status.txt')
AppContext.MODELS_INFO_FOLDER = os.path.join(AppContext.METADATA_DIR, 'trained_models_info')

AppContext.SERVER_HOST = "127.0.0.1"
AppContext.SERVER_PORT = 8080
AppContext.CLIENT_PORT = 8081

# ModelParseAndTrainTask(model_options={
#     "model_id": "1234",
#     "model_def": "[CONV(32,3,1), CONV(32,3,1), MAXPOOLING(2), CONV(64,3,1), CONV(64,3,1), DENSE(500), DENSE(100), SOFTMAX(10)]"
# }).start()
#
# ModelParseAndTrainTask(model_options={
#     "model_id": "1234",
#     "model_def": "[DENSE(500), DENSE(100), SOFTMAX(10)]"
# }).start()
#
# ModelParseAndTrainTask(model_options={
#     "model_id": "1234",
#     "model_def": "[SOFTMAX(10)]"
# }).start()
#
# ModelParseAndTrainTask(model_options={
#     "model_id": "1234",
#     "model_def": "[CONV(32,3,1), CONV(32,3,1), MAXPOOLING(2), CONV(64,3,1), CONV(64,3,1), MAXPOOLING(2), DENSE(500), DENSE(100), SOFTMAX(10)]"
# }).start()
#
# ModelParseAndTrainTask(model_options={
#     "model_id": "1234",
#     "model_def": "[CONV(32,3,1), MAXPOOLING(2), DENSE(500), SOFTMAX(10)]"
# }).start()
#
ModelParseAndTrainTask(model_options={
    "model_id": "1234",
    "model_def": "[CONV(32,3,1), DENSE(500), SOFTMAX(10)]"
}).start()
#