from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction
from Agents.Agent import Agent

class SarsaAgent(Agent):

    policy: Policy = None
    action_value_function: ActionValueFunction = None
    alpha = 0
    gamma = 0

    old_X_s = None
    old_O_s = None
    old_value = 0
    old_action = None
    iteration = 1

    def __init__(self, policy, action_value_function, alpha, gamma):
        self.policy = policy
        self.action_value_function = action_value_function
        self.alpha = alpha
        self.gamma = gamma


    def pick_action(self, X_s, O_s, reward):
        (action, value) = self.policy.pick_action(self.action_value_function, X_s, O_s, self.iteration)
        # print(action, value)

        if not (self.old_X_s is None):
            updated_value = self.old_value + self.alpha * (reward + self.gamma * value - self.old_value)
            # print(value)
            # print(updated_value)
            self.action_value_function.set_action_value(self.old_X_s, self.old_O_s, self.old_action, updated_value)

        self.old_X_s = X_s.copy()
        self.old_O_s = O_s.copy()
        self.old_value = value
        self.old_action = action
        return action

    def episode_done(self, reward, iteration):
        updated_value = self.old_value + self.alpha * (reward - self.old_value)
        self.action_value_function.set_action_value(self.old_X_s, self.old_O_s, self.old_action, updated_value)
        self.old_X_s = None
        self.old_O_s = None
        self.old_value = 0
        self.old_action = None
        self.iteration = iteration