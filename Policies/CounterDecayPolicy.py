import numpy as np

import Game
from helpers import Position
from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction
import helpers as h

class CounterDecayPolicy(Policy):

    counter = {}
    times_visited = {}
    last_visited = {}
    lambd = 0
    comma = 0.7

    def __init__(self, lambd):
        self.lambd = lambd

    def get_count(self, hash):
        # print(hash)
        try:
            return self.counter[hash]
        except:
            return 1

    def get_counts(self, hashes):
        counts = np.zeros(hashes.shape)
        it = np.nditer(hashes, flags=['multi_index', 'refs_ok'])
        for hash in it:
            hash = hash.item()
            index = it.multi_index
            count = self.get_count(hash)
            counts[index[0], index[1]] = count
        return counts

    def get_last_time_visited(self, hash):
        try:
            return self.last_visited[hash]
        except:
            return 1

    def get_last_time_visited_states(self, hashes):
        last_time_visited_states = np.zeros(hashes.shape)
        it = np.nditer(hashes, flags=['multi_index', 'refs_ok'])
        for hash in it:
            hash = hash.item()
            index = it.multi_index
            last_time_visited = self.get_last_time_visited(hash)
            last_time_visited_states[index[0], index[1]] = last_time_visited
        return last_time_visited_states

    def pick_action(self, value_fn: ActionValueFunction, X_s, O_s, iteration):

        hashed_state = Game.hash_state(X_s, O_s)

        if not (hashed_state in self.counter):
            self.counter[hashed_state] = np.ones(X_s.shape)
            self.times_visited[hashed_state] = np.zeros(X_s.shape)
            self.last_visited[hashed_state] = np.ones(X_s.shape)

        filter = np.logical_or(X_s, O_s)

        exploit_values = value_fn.get_action_values(X_s, O_s)
        decay = self.lambd**(iteration - self.last_visited[hashed_state])
        decay = decay + 0.000001
        counters = self.counter[hashed_state] * decay
        mean = np.mean(counters[counters != 1])
        explore_values = mean / (self.counter[hashed_state] * decay)
        linear_combination = (1 - self.comma) * explore_values + self.comma * exploit_values

        linear_combination[filter] = -10000
        max_value = np.max(linear_combination)
        indices = np.argwhere(linear_combination == max_value)
        # print(values)
        # print(max_value)
        index = np.random.choice(indices.shape[0])
        (y, x) = indices[index, :]
        # print(values[y, x])
        self.counter[hashed_state][y, x] += 1
        self.times_visited[hashed_state][y, x] += 1
        self.last_visited[hashed_state][y, x] = iteration
        return Position(x, y), exploit_values[y, x]
