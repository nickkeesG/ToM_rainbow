import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import numpy as np

class TomNet(keras.Model):
    def __init__(self,
                 belief_dim,
                 obs_dim,
                 num_actions,
                 atom_size = 51):
        
        super(TomNet, self).__init__()

        self.atom_size = atom_size
        
        self.zero_layer = keras.Sequential(
            [
                keras.Input(shape=(belief_dim*3 + obs_dim,)),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001), activation='relu'),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001),activation='tanh'),
            ]
        )
        
        self.char_layer = keras.Sequential(
            [
                keras.Input(shape=(belief_dim*3 + obs_dim,)),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001),activation='relu'),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001),activation='tanh'),
            ]
        )

        self.ment_layer = keras.Sequential(
            [
                keras.Input(shape=(belief_dim*3 + obs_dim,)),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001),activation='relu'),
                layers.Dense(belief_dim, kernel_regularizer=l2(0.001),activation='tanh'),
            ]
        )

        self.q_layer = keras.Sequential(
            [
                keras.Input(shape=(belief_dim*3,)),
                layers.Dense(512, kernel_regularizer=l2(0.001),activation='relu'),
                layers.Dense(num_actions*atom_size,kernel_regularizer=l2(0.001)),
                layers.Reshape((num_actions, atom_size))
            ]
        )

        self.a_layer = keras.Sequential(
            [
                keras.Input(shape=(belief_dim*2 + obs_dim,)),
                layers.Dense(512, kernel_regularizer=l2(0.001), activation='relu'),
                layers.Dense(num_actions, kernel_regularizer=l2(0.001)),
            ]
        )

    def q_call(self, b0_prev, b1c_prev, b1m_prev, observation):
        observation = tf.cast(observation, dtype=np.float32)
        
        assert(not tf.reduce_any(tf.math.is_nan(b0_prev)))
        assert(not tf.reduce_any(tf.math.is_nan(b1c_prev)))
        assert(not tf.reduce_any(tf.math.is_nan(b1m_prev)))
        b0_in = tf.concat([b0_prev, b1c_prev, b1m_prev, observation], 1)

        assert(not tf.reduce_any(tf.math.is_nan(b0_in)))
        assert(np.max(b0_in) <= 1)
        assert(np.min(b0_in) >= -1)
        b0 = self.zero_layer(b0_in)

        b1c_in = tf.concat([b0, b1c_prev, b1m_prev, observation], 1)
        b1c = self.char_layer(b1c_in)

        b1m_in = tf.concat([b0, b1c, b1m_prev, observation], 1)
        b1m = self.ment_layer(b1m_in)

        b1c_stop = tf.stop_gradient(b1c)
        b1m_stop = tf.stop_gradient(b1m)

        q_in = tf.concat([b0, b1c_stop, b1m_stop], 1)
        q = self.q_layer(q_in)
        q = tf.nn.softmax(q, axis=-1)
        q = tf.clip_by_value(q, 1e-3, 1)
        
        assert(not tf.reduce_any(tf.math.is_nan(b0)))
        assert(not tf.reduce_any(tf.math.is_nan(b1c)))
        assert(not tf.reduce_any(tf.math.is_nan(b1m)))

        return q, b0, b1c, b1m

    def full_call(self, b0_prev, b1c_prev, b1m_prev, observation):
        q, b0, b1c, b1m = self.q_call(b0_prev, b1c_prev, b1m_prev, observation)

        a_in = tf.concat([b1c, b1m, observation], 1)

        a = self.a_layer(a_in)
        a = tf.nn.softmax(a)

        return q, a, b0, b1c, b1m
