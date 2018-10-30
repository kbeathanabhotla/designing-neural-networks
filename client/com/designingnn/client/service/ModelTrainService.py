from com.designingnn.client.service.ModelParseAndTrainTask import ModelParseAndTrainTask


class ModelTrainService:
    def __init__(self):
        pass

    def train_model(self, model_options):
        ModelParseAndTrainTask(model_options).start()
