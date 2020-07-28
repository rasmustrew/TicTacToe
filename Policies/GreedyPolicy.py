import numpy as np
from helpers import Position
from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction

class GreedyPolicy(Policy):

    def pick_action(self, value_fn: ActionValueFunction, X_s, O_s, iteration):

        values = value_fn.get_action_values(X_s, O_s).to('cpu').detach().numpy().squeeze()
        filter = np.logical_or(X_s, O_s)
        values[filter] = -10000
        max_value = np.max(values)
        indices = np.argwhere(values == max_value)
        # print(values)
        # print(max_value)
        index = np.random.choice(indices.shape[0])
        (y, x) = indices[index, :]
        # print(values[y, x])
        return Position(x, y), max_value

