from hashlib import sha1
import numpy as np

import Game
from helpers import Position
from ValueFunctions.ActionValueFunction import ActionValueFunction
import helpers as h

class LookupTable(ActionValueFunction):

    lookup_table = {}
    ## { state 1: [
    #                [-2.3, 2.4, 15],
    #                [-2.3, 2.4, 15],
    #                [-2.3, 2.4, 15]
    #             ]
    # }

    def get_action_value(self, X_s, O_s, p: Position):
        hashed = Game.hash_state(X_s, O_s)
        if not (hashed in self.lookup_table):
            self.lookup_table[hashed] = np.zeros(X_s.shape)


        return self.lookup_table[hashed][p.y, p.x]

    def get_action_values(self, X_s, O_s):
        hashed = Game.hash_state(X_s, O_s)
        if not (hashed in self.lookup_table):
            self.lookup_table[hashed] = np.zeros(X_s.shape)


        actions = self.lookup_table[hashed].copy()
        # print(actions)

        return actions

    def set_action_value(self, X_s, O_s, p: Position, value):
        hashed = Game.hash_state(X_s, O_s)

        if not (hashed in self.lookup_table):
            self.lookup_table[hashed] = np.zeros(X_s.shape)
        # print(self.lookup_table[hashed])
        self.lookup_table[hashed][p.y, p.x] = value
        # print(self.lookup_table[hashed])
        # print(value)
        # print(self.times_visited[hashed])

    def set_action_values(self, X_s, O_s, values):
        hashed = Game.hash_state(X_s, O_s)

        if not (hashed in self.lookup_table):
            self.lookup_table[hashed] = np.zeros(X_s.shape)
        # print(self.lookup_table[hashed])
        self.lookup_table[hashed] = values

    def save(self, path):
        np.savez('objects/' + path + '.npz', table=self.lookup_table)

    def load(self, path):
        res = np.load('objects/' + path + '.npz', allow_pickle=True)
        self.lookup_table = res['table']




