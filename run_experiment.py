import numpy as np

import tom_agent
#import gin.tf
from hanabi_learning_environment import rl_env

class ObservationStacker():
    def __init__(self, history_size, observation_size, num_players):
        self._history_size = history_size
        self._observation_size = observation_size
        self._num_players = num_players
        self._obs_stacks = list()
        for _ in range(self._num_players):
            self._obs_stacks.append(np.zeros(self._observation_size * self._history_size))

    def add_observation(self, observation, current_player):
        self._obs_stacks[current_player] = np.roll(self._obs_stacks[current_player],
                                                    -self._observation_size)
        self._obs_stacks[current_player][(self._history_size - 1) *
                                         self._observation_size:] = observation

    def get_observation_stacks(self, current_player):
        return self._obs_stacks[current_player]

    def observation_size(self):
        return self._observation_size * self._history_size

    def reset_stack(self):
        for i in range(self._num_players):
            self._obs_stacks[i].fill(0.0)

def load_gin_configs(gin_file):
    gin.parse_config_file(gin_file)

def create_environment(game_type='Hanabi-Full', num_players=2):
    return rl_env.make(
        environment_name=game_type, num_players=num_players, pyhanabi_path=None)

def create_obs_stacker(environment, history_size=4):
    return ObservationStacker(history_size,
                        environment.vectorized_observation_shape()[0],
                        environment.players)

def create_agent(environment, obs_stacker):
    return tom_agent.ToMAgent(observation_size=obs_stacker.observation_size(),
                              num_actions=environment.num_moves(),
                              num_players=environment.players)

def format_legal_moves(legal_moves, action_dim):
    new_legal_moves = np.full(action_dim, -float('inf'))
    if legal_moves:
        new_legal_moves[legal_moves] = 0
    return new_legal_moves

def parse_observations(observations, num_actions, obs_stacker):
    current_player = observations['current_player']
    current_player_observation = (
            observations['player_observations'][current_player])

    legal_moves = current_player_observation['legal_moves_as_int']
    legal_moves = format_legal_moves(legal_moves, num_actions)

    observation_vector = current_player_observation['vectorized']
    obs_stacker.add_observation(observation_vector, current_player)
    observation_vector = obs_stacker.get_observation_stacks(current_player)

    return current_player, legal_moves, observation_vector
    

def run_one_episode(agent, environment, obs_stacker):

    obs_stacker.reset_stack()
    observations = environment.reset()
    current_player, legal_moves, observation_vector = (
            parse_observations(observations, environment.num_moves(), obs_stacker))

    beliefs = list()
    for _ in range(environment.players):
        beliefs.append({"zero":np.zeros((1, agent.belief_size), dtype=np.float32),
                       "char":np.zeros((1, agent.belief_size), dtype=np.float32),
                       "ment":np.zeros((1, agent.belief_size), dtype=np.float32)})
    
    action, beliefs[current_player] = agent.step(0, current_player, legal_moves, observation_vector, 
                                                 beliefs[current_player], begin=True)

    is_done = False
    total_reward = 0
    step_number = 0
    
    has_played = {current_player}

    reward_since_last_action = np.zeros(environment.players)

    while not is_done:
        observations, reward, is_done, _ = environment.step(action)

        total_reward += reward
        reward_since_last_action += reward

        step_number += 1
        if is_done:
            break
        current_player, legal_moves, observation_vector = (
                parse_observations(observations, environment.num_moves(), obs_stacker))

        if current_player in has_played:
            action, beliefs[current_player] = agent.step(reward_since_last_action[current_player], current_player, 
                                                         legal_moves, observation_vector, beliefs[current_player])
        else:
            action, beliefs[current_player] = agent.step(0, current_player, legal_moves, observation_vector,
                                                         beliefs[current_player], begin=True)
            has_played.add(current_player)

        reward_since_last_action[current_player] = 0

    agent.end_episode(reward_since_last_action)

    return step_number, total_reward

def run_one_phase(agent, environment, obs_stacker, min_steps, run_mode_str):
    step_count = 0
    num_episodes = 0
    sum_returns = 0
    
    while step_count < min_steps:
        episode_length, episode_return = run_one_episode(agent, environment, obs_stacker)
        step_count += episode_length
        sum_returns += episode_return
        num_episodes += 1

    return step_count, sum_returns, num_episodes

def run_one_iteration(agent, environment, obs_stacker, training_steps):
    number_steps, sum_returns, num_episodes = (run_one_phase(agent, environment, obs_stacker, training_steps, 'train'))
    print(sum_returns / num_episodes)

def run_experiment(agent, environment, obs_stacker,
                   num_iterations=100000, training_steps=1000):
    for iteration in range(num_iterations):
        run_one_iteration(agent, environment, obs_stacker, training_steps)
