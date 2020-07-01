import numpy as np
from Agents.SarsaAgent import SarsaAgent
from helpers import Player, Position


def print_game_state(X_s, O_s):

    translate_to_symbol = np.array(['-', 'O', 'X'])
    translator = lambda p: translate_to_symbol[p]

    game_array = X_s * 2 + O_s

    printable = translator(game_array)
    print(printable)
    print()

def hash_state(X_s, O_s):
    combined = X_s * 2 + O_s
    hashed = [str(i) for i in combined.flatten()]
    hashed = ''.join(hashed)
    return hashed

def get_hash_for_action(X_s, O_s, p: Position):
    X_s = X_s.copy()
    O_s = O_s.copy()
    X_s[p.y, p.x] = 1
    return hash_state(X_s, O_s)



def get_hashes_for_actions(X_s, O_s):
    hashes = np.empty(X_s.shape, 'object')
    combined = X_s * 2 + O_s
    it = np.nditer(combined, flags=['multi_index'])
    for x in it:
        index = it.multi_index
        if (x == 0):
            p = Position(index[1], index[0])
            action_hash = get_hash_for_action(X_s, O_s, p)
            hashes[index[0], index[1]] = action_hash
        else:
            hashes[index[0], index[1]] = '000000000'
    return hashes

class Game:

    player_agent: SarsaAgent = None
    opponent_agent: SarsaAgent = None

    def __init__(self, player_agent, opponent_agent):
        self.player_agent = player_agent
        self.opponent_agent = opponent_agent
        self.X_s = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.int)

        self.O_s = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.int)


    X_s = None
    O_s = None

    def is_game_won(self, x_or_o_state):
        eye = np.eye(x_or_o_state.shape[0])
        won_row = np.any(np.sum(x_or_o_state, axis=0) == 3)
        won_col = np.any(np.sum(x_or_o_state, axis=1) == 3)
        won_diag = np.array_equal(np.logical_and(x_or_o_state, eye), eye)
        won_diag_2 = np.array_equal(np.logical_and(x_or_o_state, np.rot90(eye)), np.rot90(eye))
        won = won_row or won_col or won_diag or won_diag_2
        return won

    def did_player_win(self):
        return self.is_game_won(self.X_s)

    def did_player_lose(self):
        return self.is_game_won(self.O_s)

    def is_draw(self):
        return np.all(np.logical_or(self.X_s, self.O_s) == 1)

    def place_piece(self, player: Player, p: Position):
        if player == Player.X:
            self.X_s[p.y, p.x] = 1
        elif player == Player.O:
            self.O_s[p.y, p.x] = 1



    reward = 0
    def do_timestep(self, iteration):
        self.do_player_action(self.reward)
        if self.did_player_win():
            self.player_agent.episode_done(reward=2, iteration=iteration)
            return 2
        elif self.is_draw():
            self.player_agent.episode_done(reward=-1, iteration=iteration)
            return -1
        self.do_opponent_action()
        if self.did_player_lose():
            self.player_agent.episode_done(reward=-2, iteration=iteration)
            return -2
        elif self.is_draw():
            self.player_agent.episode_done(reward=-1, iteration=iteration)
            return -1
        return 0

    def do_player_action(self, reward):
        player_action = self.player_agent.pick_action(self.X_s, self.O_s, reward)
        self.place_piece(Player.X, player_action)
        return player_action

    def do_opponent_action(self):
        opponent_action = self.opponent_agent.pick_action(self.O_s,
                                                          self.X_s, 0)  # Inverted since opponent think they are X_s
        self.place_piece(Player.O, opponent_action)


