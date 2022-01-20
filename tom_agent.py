import numpy as np
import random
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tf_agents.agents.categorical_dqn.categorical_dqn_agent import project_distribution

from replay_buffer import ReplayBuffer
from arch import TomNet

Transition = collections.namedtuple(
        'Transition', ['reward', 'observation', 'belief', 'legal_acts', 'action', 'begin'])

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):

    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus

class ToMAgent():
    
    def __init__(self,
                 belief_size=256,
                 num_actions=None,
                 observation_size=None,
                 num_players=None,
                 min_replay_history=500, #misleading name please change
                 target_update_period=500,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_train=0.02,
                 epsilon_decay_period=1000,):
        self.belief_size = belief_size
        self.num_actions = num_actions
        self.observation_size = observation_size
        self.num_players = num_players
        self.min_replay_history = min_replay_history #misleading name please change
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_train = epsilon_train
        self.epsilon_decay_period = epsilon_decay_period
        self.training_steps = 0

        self.gamma = 0.99
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0001 ,clipnorm = 1.0)

        self.active_net = TomNet(belief_size, observation_size, num_actions)
        self.target_net = TomNet(belief_size, observation_size, num_actions)
        self.target_net.set_weights(self.active_net.get_weights())

        self.memory = ReplayBuffer(observation_size, belief_size, num_actions, 10000)

        self.support = tf.cast(tf.linspace(-25, 25, 51), dtype=tf.float32)

        self.transitions = [[] for _ in range(num_players)]

    def step(self, reward, current_player, legal_actions, observation, belief, begin=False):

        self._train_step()

        self.action, new_belief = self._select_action(observation, belief, legal_actions)
        self._record_transition(current_player, reward, observation, belief, legal_actions, self.action, begin)

        return self.action, new_belief

    def end_episode(self, final_rewards):
        self._post_transitions(terminal_rewards=final_rewards)

    def _record_transition(self, current_player, reward, observation, belief, legal_acts, action, begin):
        self.transitions[current_player].append(
                Transition(reward, observation, belief, legal_acts, action, begin))

    def _post_transitions(self, terminal_rewards):
        for player in range(self.num_players):
            num_transitions = len(self.transitions[player])

            for index, transition in enumerate(self.transitions[player]):
                final_transition = index == num_transitions - 1

                #gonna be janky, I'm sorry...
                if player == 0:
                    offset = 0
                if player == 1:
                    offset = 1
                if len(self.transitions[(player + 1) % 2]) > index + offset:
                    partner_act = self.transitions[(player+1)%2][index+offset].action
                else:
                    partner_act = -1

                if final_transition:
                    reward = terminal_rewards[player]
                    self.memory.store(transition.observation, transition.belief,
                                      transition.action, reward,
                                      None, None, [], partner_act, True)
                else:
                    next_transition = self.transitions[player][index+1]
                    reward = next_transition.reward
                    self.memory.store(transition.observation, transition.belief,
                                      transition.action, reward,
                                      next_transition.observation,
                                      next_transition.belief,
                                      next_transition.legal_acts, 
                                      partner_act, False)

        self.transitions[0] = []
        self.transitions[1] = []

    def _select_action(self, observation, belief, legal_actions):
        epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                  self.min_replay_history, self.epsilon_train)
        
        obs_input = tf.expand_dims(observation, 0)
        dist, b0, b1c, b1m = self.active_net.q_call(belief['zero'], belief['char'], belief['ment'], obs_input) 
        q = tf.reduce_sum(dist * self.support, axis=2)

        if random.random() <= epsilon:
            legal_action_indices = np.where(legal_actions == 0.0)
            action = np.random.choice(legal_action_indices[0])
        else:
            action = tf.argmax(q + legal_actions, axis=1)[0]

        action = int(action)
            
        assert(legal_actions[action] == 0.0)
        return action, dict({'zero':b0, 'char':b1c, 'ment':b1m})

    def _train_step(self):
        if len(self.memory) > 32:
            samples = self.memory.sample_batch()
            obs = samples['obs']
            b0 = samples['b0']
            b1c = samples['b1c']
            b1m = samples['b1m']
            next_obs = samples['next_obs']
            next_b0 = samples['next_b0']
            next_b1c = samples['next_b1c']
            next_b1m = samples['next_b1m']
            legal_acts = samples['legal_acts']
            actions = samples['acts']
            rewards = samples['rews']
            done = samples['done']
            partner_acts = samples['partner_acts']

            legal_acts = tf.cast(legal_acts, dtype=tf.float32)

            q_dist, _, _, _ = self.active_net.q_call(next_b0, next_b1c, next_b1m, next_obs) 
            q_values = tf.reduce_sum(q_dist * self.support, axis=2)
            q_values = tf.add(q_values, legal_acts)

            next_action = tf.argmax(q_values, axis=1)

            next_dist, _, target_char, _ = self.target_net.q_call(next_b0, next_b1c, next_b1m, next_obs)
            next_dist = tf.gather(next_dist, next_action, axis=1, batch_dims=1)
            
            rewards = keras.layers.Reshape((-1,))(rewards)
            done = keras.layers.Reshape((-1,))(done)

            t_z = rewards + (1 - done) * self.gamma * self.support
            t_z = tf.clip_by_value(t_z, -25, 25)

            proj_dist = project_distribution(t_z, next_dist, self.support)

            masks = tf.one_hot(actions, self.num_actions)
            masks = tf.expand_dims(masks, axis=-1)

            partner_acted = tf.where(partner_acts != -1, True, False)
            partner_acts = tf.boolean_mask(partner_acts, partner_acted)
            partner_acts = tf.one_hot(partner_acts, self.num_actions)

            not_done = tf.squeeze(tf.where(done != 1, True, False))
            target_char = tf.boolean_mask(target_char, not_done)
            
            with tf.GradientTape() as tape:
                dist, a, _, char, _ = self.active_net.full_call(b0, b1c, b1m, obs)
                dist = tf.reduce_sum(tf.multiply(dist, masks), axis=1)
                loss1 = keras.losses.CategoricalCrossentropy()(proj_dist, dist)

                a = tf.boolean_mask(a, partner_acted)
                loss2 = keras.losses.CategoricalCrossentropy()(partner_acts, a)

                char = tf.boolean_mask(char, not_done)
                loss3 = keras.losses.MeanSquaredError()(target_char, char)

                total_loss = loss1 + (loss2/10) + (loss3/100)

            grads = tape.gradient(total_loss, self.active_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.active_net.trainable_variables))


        if self.training_steps % self.target_update_period == 0:
            self.target_net.set_weights(self.active_net.get_weights())

        self.training_steps += 1


