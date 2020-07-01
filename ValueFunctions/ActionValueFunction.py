from abc import ABC, abstractmethod
from helpers import Position


class ActionValueFunction(ABC):

    @abstractmethod
    def get_action_value(self, X_s, O_s, p: Position):
        pass

    @abstractmethod
    def get_action_values(self, X_s, O_s):
       pass

    @abstractmethod
    def set_action_value(self, X_s, O_s, p: Position, value):
        pass

    @abstractmethod
    def set_action_values(self, X_s, O_s, values):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
