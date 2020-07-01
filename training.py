import copy

import numpy as np

from Agents import Agent
from Agents.BackwardSarsaLambdaAgent import BackwardSarsaLambdaAgent
from Agents.BareAgent import BareAgent
from Game import Game
from Policies.GreedyPolicy import GreedyPolicy
from Policies.HumanPolicy import HumanPolicy
from Policies.RandomPolicy import RandomPolicy
from ValueFunctions.LookupTable import LookupTable


def train(player_agent: Agent, opponent_agent, games, validate_each, validate_games):
    player_policy_val = GreedyPolicy()
    player_agent_val = BareAgent(player_policy_val, player_agent.action_value_function)

    opponent_agents = [opponent_agent]
    for i in range(1, games + 1):
        if i % 50000 == 0:
            opponent_policy = GreedyPolicy()
            opponent_value_function = copy.deepcopy(player_agent.action_value_function)
            opponent_agent = BareAgent(opponent_policy, opponent_value_function)
            opponent_agents.append(opponent_agent)

        if (i % validate_each) == 0:
            print("iteration: ", i)
            validate(player_agent_val, opponent_agents, validate_games)

        # print("iteration: ", i)
        current_opponent = np.random.choice(opponent_agents)
        game = Game(player_agent, current_opponent)

        result = play_game(game, i)
    return opponent_agents


def validate(player_agent, opponent_agents, games):

    record = {
    }
    for j, opponent in enumerate(opponent_agents):

        record[j] = {-2: 0,
                     -1: 0,
                     2: 0}

        for i in range(1, games):
            result = play_game(Game(player_agent, opponent), i)
            record[j][result] += 1

        win_percent = record[j][2] / games
        draw_percent = record[j][-1] / games
        loss_percent = record[j][-2] / games

        print("opponent: ", j)
        print("win percent: ", win_percent * 100)
        print("draw percent: ", draw_percent * 100)
        print('loss percent: ', loss_percent * 100)


def play_human(opponent_agent, player_first):
    player_policy = HumanPolicy()

    # player_policy = RandomPolicy()
    player_agent = BareAgent(player_policy, None)

    game = Game(player_agent, opponent_agent)

    if not player_first:
        game.do_opponent_action()
    play_game(game, 0)


def play_game(game: Game, iteration):
    coin_flip = np.random.random()
    if coin_flip > 0.5:  # The opponent starts
        game.do_opponent_action()
        # print("player 2")
    # else:
        # print("player 1")

    # game.print_game_state()

    result = 0
    while result == 0:
        result = game.do_timestep(iteration)
        # game.print_game_state()

    # if result == -2:
    #     print("Player lost")
    # elif result == -1:
    #     print("Draw")
    # else:
    #     print("Player won")
    return result