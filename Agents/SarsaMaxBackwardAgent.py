import Game
from Policies.Policy import Policy
from Policies.GreedyPolicy import GreedyPolicy
from Policies.OneOverKGreedyPolicy import OneOverKGreedyPolicy
from Policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy
from ValueFunctions.ActionValueFunction import ActionValueFunction
from Agents.Agent import Agent
import numpy as np
import helpers as h

class SarsaMaxBackwardAgent(Agent):

    greedy_policy: Policy = GreedyPolicy()
    epsilon_greedy_policy: Policy = EpsilonGreedyPolicy(0.2)
    action_value_function: ActionValueFunction = None
    alpha = 0
    gamma = 0
    lambd = 0

    old_X_s = None
    old_O_s = None
    old_value = 0
    old_action = None
    iteration = 1
    E = {}
    states = {}


    def __init__(self, action_value_function, alpha, gamma, lambd):
        self.action_value_function = action_value_function
        # self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd
        self.alpha = alpha

    def pick_action(self, X_s, O_s, reward):
        (action, value) = self.epsilon_greedy_policy.pick_action(self.action_value_function, X_s, O_s, self.iteration)
        # print(action, value)
        hashed_state = Game.hash_state(X_s, O_s)

        values = self.action_value_function.get_action_values(X_s, O_s)

        TD_error = 0

        if not (self.old_X_s is None):
            TD_error = reward + self.gamma * np.max(values) - self.old_value

        for k, v in self.E.items():
            self.E[k] = self.gamma * self.lambd * v
            (E_X_s, E_O_s) = self.states[k]
            old_Values = self.action_value_function.get_action_values(E_X_s, E_O_s)
            updated_values = old_Values + self.alpha * TD_error * self.E[k]
            self.action_value_function.set_action_values(E_X_s, E_O_s, updated_values)

        if not (hashed_state in self.E):
            self.E[hashed_state] = np.zeros(X_s.shape)
            self.states[hashed_state] = (X_s.copy(), O_s.copy())
        self.E[hashed_state][action.y, action.x] += 1

        self.old_X_s = X_s.copy()
        self.old_O_s = O_s.copy()
        self.old_value = value
        self.old_action = action
        return action

    def episode_done(self, reward, iteration):
        # updated_value = self.old_value + self.alpha * (reward + self.gamma * 1 - self.old_value)
        # self.action_value_function.set_action_value(self.old_X_s, self.old_O_s, self.old_action, updated_value)
        TD_error = reward - self.old_value
        for k, v in self.E.items():
            self.E[k] = self.gamma * self.lambd * v
            (E_X_s, E_O_s) = self.states[k]
            updated_values = self.action_value_function.get_action_values(E_X_s, E_O_s) + self.alpha * TD_error * self.E[k]
            self.action_value_function.set_action_values(E_X_s, E_O_s, updated_values)

        self.old_X_s = None
        self.old_O_s = None
        self.old_value = 0
        self.old_action = None
        self.iteration = iteration

        if iteration % 1000:
            self.alpha = self.alpha / 1.1

        # if iteration < 100000:
        #     self.alpha = 0.9
        # elif iteration > 100000 and iteration < 200000:
        #     self.alpha = 0.5
        # elif iteration > 200000 and iteration < 300000:
        #     self.alpha = 0.09
        # elif iteration > 300000 and iteration < 400000:
        #     self.alpha = 0.05
        # elif iteration > 400000 and iteration < 500000:
        #     self.alpha = 0.009
        # elif iteration > 500000 and iteration < 600000:
        #     self.alpha = 0.005
        # elif iteration > 600000 and iteration < 700000:
        #     self.alpha = 0.0009
        # elif iteration > 700000 and iteration < 800000:
        #     self.alpha = 0.0005
        # elif iteration > 800000 and iteration < 900000:
        #     self.alpha = 0.00009
        # elif iteration > 900000 and iteration < 1000000:
        #     self.alpha = 0.00005
        self.states = {}
        self.E = {}

