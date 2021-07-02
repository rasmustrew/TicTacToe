import pickle

import helpers
from Agents.BackwardSarsaLambdaAgent import BackwardSarsaLambdaAgent
from Agents.BareAgent import BareAgent
from Policies.CounterDecayPolicy import CounterDecayPolicy
from Policies.RandomPolicy import RandomPolicy
from Policies.UCBPolicy import UCBPolicy
from ValueFunctions.CNN import CNN
from ValueFunctions.LookupTable import LookupTable
from training import train, validate, play_human
from Policies.GreedyPolicy import GreedyPolicy


player_policy_train = UCBPolicy(0.2)
player_value_function = CNN(gamma=0.9, lambd=0.8, alpha=0.5)
player_agent_train = BackwardSarsaLambdaAgent(player_policy_train, player_value_function, 0.9, 0.9, 0.8)
opponent_policy = RandomPolicy()
opponent_value_function = None
opponent_agent = BareAgent(opponent_policy, opponent_value_function)

opponent_agents = train(player_agent_train, opponent_agent, 2001, 1000, 500)
# player_value_function.save('SARSA_MAX_backward__UCB_vs_Self')
print("ended training!")


player_policy_val = GreedyPolicy()
player_agent_val = BareAgent(player_policy_val, player_value_function)

# opponent_policy = RandomPolicy()
# opponent_value_function = None
# opponent_agent = BareAgent(opponent_policy, opponent_value_function)
# opponent_agents = [opponent_agent]

validate(player_agent_val, opponent_agents, 501)
# ## ------------------------------- ####
#
#
# print("ended validation")
## ------------------------------- ####

# # opponent_policy = RandomPolicy()
# opponent_policy = GreedyPolicy()
# opponent_value_function = LookupTable()
# opponent_value_function.load('SARSA_MAX_backward__UCB_vs_Self')
# opponent_agent = BareAgent(opponent_policy, opponent_value_function)
#
# while True:
#     play_human(opponent_agent, True)

## ------------------------------- ####

