from enum import Enum


class Player(Enum):
    X = 1
    O = 2


class Position:

    x, y = 0, 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return str((self.x, self.y))


