import numpy as np

import Game
from helpers import Position
from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction
import helpers as h

class OneOverKGreedyPolicy(Policy):
    times_visited = {}

    random_actions = 0


    def random_action(self, X_s, O_s):
        random_values = np.random.rand(*X_s.shape)
        filter = np.logical_or(X_s, O_s)
        random_values[filter] = -10000
        (y, x) = np.unravel_index(random_values.argmax(), random_values.shape)
        return Position(x, y)


    def pick_action(self, value_fn: ActionValueFunction, X_s, O_s, iteration):

        hashed_state = Game.hash_state(X_s, O_s)

        if not (hashed_state in self.times_visited):
            self.times_visited[hashed_state] = np.zeros(X_s.shape)


        epsilon = (1 / np.sqrt(iteration))
        rand_n = np.random.random()
        # print("epsilon: ", epsilon)
        if rand_n < epsilon:
            self.random_actions += 1
            # print("random action")
            random_action = self.random_action(X_s, O_s)
            self.times_visited[hashed_state][random_action.y, random_action.x] += 1
            return random_action, value_fn.get_action_value(X_s, O_s, random_action)

        values = value_fn.get_action_values(X_s, O_s)
        filter = np.logical_or(X_s, O_s)
        values[filter] = -10000
        max_value = np.max(values)
        indices = np.argwhere(values == max_value)
        # print(values)
        # print(max_value)
        index = np.random.choice(indices.shape[0])
        (y, x) = indices[index, :]
        # print(values[y, x])
        self.times_visited[hashed_state][y, x] += 1
        return Position(x, y), max_value



