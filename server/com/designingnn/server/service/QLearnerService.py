class QLearnerService:
    def __init__(self):
        pass

    def generate_new_model(self):
        return "[CONV(32,3,1), CONV(32,3,1), MAXPOOLING(2), CONV(64,3,1), CONV(64,3,1), DENSE(500), DENSE(100), SOFTMAX(10)]"
