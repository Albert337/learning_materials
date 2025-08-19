import torch
import torch.nn as nn
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on:",device)

class DQN(nn.Module):
    def __init__(self,n_input,n_output):
        super(DQN, self).__init__()
        self.n_input=n_input
        self.n_output=n_output
        self.model=nn.Sequential(
            nn.Linear(self.n_input,128),
            nn.ReLU(),
            nn.Linear(128,self.n_output),
        )

    def forward(self,x):
        return self.model(x)

    def act(self,obs):
        obs_tensor=torch.tensor(obs,dtype=torch.float32)
        q_value=self.forward(obs_tensor.unsqueeze(0))
        max_q_idx=torch.argmax(q_value)
        action=max_q_idx.detach().item()

        return action


class Replaymemory:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.MEMORY_SIZE = 10000
        self.BATCH_SIZE = 64

        self.all_s = np.empty((self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.all_a = np.random.randint(0, n_a, size=self.MEMORY_SIZE, dtype=np.int8)
        self.all_r = np.empty((self.MEMORY_SIZE, 1), dtype=np.float32)
        self.all_done = np.random.randint(0, 2, size=self.MEMORY_SIZE, dtype=np.int8)
        self.all_s_ = np.empty((self.MEMORY_SIZE, self.n_s), dtype=np.float32)
        self.t_memo = 0
        self.t_max = 0

    def add_memo(self, state, action, reward, next_state, done):
        self.all_s[self.t_memo] = state
        self.all_a[self.t_memo] = action
        self.all_r[self.t_memo] = reward
        self.all_done[self.t_memo] = done
        self.all_s_[self.t_memo] = next_state
        self.t_max = max(self.t_max, self.t_memo + 1)
        self.t_memo = (self.t_memo + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.t_max >= self.BATCH_SIZE:
            # indices = np.random.sample(range(0, self.t_max), self.BATCH_SIZE, replace=False)
            indices = np.random.choice(self.t_max, self.BATCH_SIZE, replace=False)

        else:
            indices = range(0, self.t_max)

        batch_s = self.all_s[indices]
        batch_a = self.all_a[indices]
        batch_r = self.all_r[indices]
        batch_done = self.all_done[indices]
        batch_s_ = self.all_s_[indices]

        batch_s_tensor = torch.as_tensor(batch_s, dtype=torch.float32) 
        batch_a_tensor = torch.as_tensor(batch_a, dtype=torch.int64).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(batch_r, dtype=torch.float32).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(batch_done, dtype=torch.float32).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(batch_s_, dtype=torch.float32)

        return batch_s_tensor, batch_a_tensor, batch_r_tensor, batch_done_tensor, batch_s__tensor


class DQNAgent():
    def __init__(self, n_inputs, n_outputs):
        super(DPOAgent, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.gamma = 0.99
        self.learning_rate = 1e-3

        self.memo = Replaymemory(self.n_inputs, self.n_outputs)

        self.online_net = DQN(self.n_inputs, self.n_outputs)
        self.target_net = DQN(self.n_inputs, self.n_outputs)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

    def update(self):
        pass

