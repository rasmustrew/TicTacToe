from abc import ABC, abstractmethod

from ValueFunctions import ActionValueFunction


class Agent(ABC):

    action_value_function: ActionValueFunction = None

    @abstractmethod
    def pick_action(self, X_s, O_s, reward):
        pass

    @abstractmethod
    def episode_done(self, reward, iteration):
        pass

    # @abstractmethod
    # def get_value_function(self):
    #     pass