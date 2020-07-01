from Policies.Policy import Policy
from ValueFunctions.ActionValueFunction import ActionValueFunction
from Agents.Agent import Agent

class BareAgent(Agent):

    policy: Policy = None
    action_value_function: ActionValueFunction = None


    def __init__(self, policy, action_value_function):
        self.policy = policy
        self.action_value_function = action_value_function

    def pick_action(self, X_s, O_s, reward):
        (action, value) = self.policy.pick_action(self.action_value_function, X_s, O_s, 1)
        return action

    def episode_done(self, reward, iteration):
        pass