import numpy as np
from helpers import Position
from Policies.Policy import Policy

class RandomPolicy(Policy):

    def pick_action(self, value_fn, X_s, O_s, iteration):
        random_values = np.random.rand(*X_s.shape)
        filter = np.logical_or(X_s, O_s)
        random_values[filter] = 0
        (y, x) = np.unravel_index(random_values.argmax(), random_values.shape)
        return Position(x, y), random_values[y, x]
