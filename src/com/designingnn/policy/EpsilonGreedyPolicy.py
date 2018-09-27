import numpy as np

from com.designingnn.policy.AbstractPolicy import AbstractPolicy
from com.designingnn.resources import AppConfig


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, eps=AppConfig.DEFAULT_EPSILON):
        super(EpsilonGreedyPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        config = super(EpsilonGreedyPolicy, self).get_config()
        config['eps'] = self.eps
        return config
