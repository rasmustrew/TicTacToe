import numpy as np

import Game
from helpers import Position
from Game import hash_state
from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction

class HumanPolicy(Policy):

    def is_move_legal(self, X_S, O_S, p: Position):
        board = np.logical_or(X_S, O_S)
        return board[p.y, p.x] == 0

    def pick_action(self, value_fn: ActionValueFunction, X_s, O_s, iteration):

        has_picked_legal_move = False
        p = None

        while not has_picked_legal_move:

            print("board: ")
            Game.print_game_state(X_s, O_s)
            player_input = input("Your move:") #1,1 or 2,0 etc.

            coords = player_input.split(",")
            x = coords[0]
            y = coords[1]
            p = Position(int(x), int(y))

            has_picked_legal_move = self.is_move_legal(X_s, O_s, p)

        return p, 0

