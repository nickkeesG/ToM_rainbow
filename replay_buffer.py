import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, belief_dim, act_dim, size, batch_size = 32):
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.act_dim = act_dim

        self.obs_buf = np.zeros([size, obs_dim], dtype = np.float32)     
        self.next_obs_buf = np.zeros([size, obs_dim], dtype = np.float32)     
        self.b0_buf = np.zeros([size, belief_dim], dtype = np.float32)
        self.b1c_buf = np.zeros([size, belief_dim], dtype = np.float32)
        self.b1m_buf = np.zeros([size, belief_dim], dtype = np.float32)
        self.next_b0_buf = np.zeros([size, belief_dim], dtype = np.float32)
        self.next_b1c_buf = np.zeros([size, belief_dim], dtype = np.float32)
        self.next_b1m_buf = np.zeros([size, belief_dim], dtype = np.float32)

        self.legal_acts_buf = np.zeros([size, act_dim], dtype = np.int32)
        self.acts_buf = np.zeros([size], dtype = np.int32)
        self.partner_acts_buf = np.zeros([size], dtype = np.int32)
        self.rews_buf = np.zeros([size], dtype = np.float32)
        self.done_buf = np.zeros(size, dtype = np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(self, obs, belief, act, rew, next_obs, next_belief, legal_acts, partner_act, done):
        self.obs_buf[self.ptr] = obs
        self.b0_buf[self.ptr] = belief['zero']
        self.b1c_buf[self.ptr] = belief['char']
        self.b1m_buf[self.ptr] = belief['ment']
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.acts_buf[self.ptr] = act
        self.partner_acts_buf[self.ptr] = partner_act

        if done:
            self.next_obs_buf[self.ptr] = np.zeros(self.obs_dim)
            self.next_b0_buf[self.ptr] = np.zeros(self.belief_dim)
            self.next_b1c_buf[self.ptr] = np.zeros(self.belief_dim)
            self.next_b1m_buf[self.ptr] = np.zeros(self.belief_dim)
            self.legal_acts_buf[self.ptr] = np.zeros(self.act_dim)
        else:
            self.next_obs_buf[self.ptr] = next_obs
            self.next_b0_buf[self.ptr] = next_belief['zero']
            self.next_b1c_buf[self.ptr] = next_belief['char']
            self.next_b1m_buf[self.ptr] = next_belief['ment']
            self.legal_acts_buf[self.ptr] = legal_acts
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs = self.obs_buf[idxs],
                    b0 = self.b0_buf[idxs],
                    b1c = self.b1c_buf[idxs],
                    b1m = self.b1m_buf[idxs],
                    next_obs = self.next_obs_buf[idxs],
                    next_b0 = self.b0_buf[idxs],
                    next_b1c = self.b1c_buf[idxs],
                    next_b1m = self.b1m_buf[idxs],
                    legal_acts = self.legal_acts_buf[idxs],
                    acts = self.acts_buf[idxs],
                    partner_acts = self.partner_acts_buf[idxs],
                    rews = self.rews_buf[idxs],
                    done = self.done_buf[idxs])
        return None

    def __len__(self):
        return self.size

        
