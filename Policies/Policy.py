from abc import ABC, abstractmethod

class Policy(ABC):

    @abstractmethod
    def pick_action(self, value_fn, X_s, O_s, iteration):
        pass