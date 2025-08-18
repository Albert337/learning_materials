from torch import nn
import torch
import numpy as np


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("computing device: ", device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_dims=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc_mean = nn.Linear(hidden_dims, action_dim)
        self.fc_std = nn.Linear(hidden_dims, action_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x))*2  ##范围在-2 到 2之间
        std = self.softplus(self.fc_std(x))+1e-3 
        return mean, std
    
    def select_action(self, x):
        with torch.no_grad():
            mu, sigma = self.forward(x)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample()
            action = action.clamp(-2,2)
        return action
    
class Critic(nn.Module):
    def __init__(self,state_dim,hidden_dim=256):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class ReplayMemory:
    def __init__(self,batch_size):
        super(ReplayMemory,self).__init__()
        self.state_cap=[]
        self.action_cap=[]
        self.value_cap=[]
        self.done_cap=[]
        self.reward_cap=[]
        self.batch_size=batch_size

    def add_memo(self,state,action,value,done,reward):
        self.state_cap.append(state)    
        self.action_cap.append(action)  
        self.value_cap.append(value)
        self.done_cap.append(done)
        self.reward_cap.append(reward)

    def sample(self):
        num_state = len(self.state_cap)
        batch_start_points = np.arange(0, num_state, self.batch_size)
        memory_indices = np.arange(num_state, dtype=np.int32)
        np.random.shuffle(memory_indices)
        batches = [memory_indices[start:start+self.batch_size] for start in batch_start_points]

        return (np.array(self.state_cap, dtype=np.float32),
                np.array(self.action_cap, dtype=np.float32),
                np.array(self.reward_cap, dtype=np.float32),
                np.array(self.value_cap,  dtype=np.float32),
                np.array(self.done_cap,   dtype=np.float32),
                batches)



    def clear_memo(self):
        self.action_cap.clear() 
        self.value_cap.clear()
        self.done_cap.clear()
        self.reward_cap.clear()
        self.state_cap.clear()


class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size, gamma=0.95, lamada=0.95,epoch=10,lr=1e-3,eposilon_clip=0.15):
        super(PPOAgent,self).__init__()

        self.lr_actor = lr
        self.lr_critic = lr
        self.gamma = gamma
        self.lamada = lamada
        self.epoch = epoch
        self.eposilon_clip = eposilon_clip

        self.actor = Actor(state_dim, action_dim).to(device)
        self.old_actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr*0.1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayMemory(batch_size)
    
    def get_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = self.actor.select_action(state)          # tensor float32
        value  = self.critic(state)                       # tensor float32
        # 保证 env.step 的输入是 (1,) 的 numpy float32
        return action.detach().cpu().numpy().astype(np.float32).reshape(-1), \
            value.detach().cpu().numpy().astype(np.float32)


    def update(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        for _ in range(self.epoch):
            memo_state, memo_action, memo_reward, memo_value, memo_done, batches = self.replay_buffer.sample()
            T = len(memo_reward)

            # ----- 计算 GAE -----
            memo_advantage = np.zeros(T, dtype=np.float32)
            discount = 1.0
            for t in range(T):
                a_t = 0.0
                discount = 1.0
                for k in range(t, T - 1):
                    delta = memo_reward[k] + self.gamma * memo_value[k + 1] * (1.0 - memo_done[k]) - memo_value[k]
                    a_t += discount * delta
                    discount *= self.gamma * self.lamada
                memo_advantage[t] = a_t

            # ----- 转 tensor（float32）-----
            adv  = torch.as_tensor(memo_advantage, dtype=torch.float32, device=device).unsqueeze(-1)
            vals = torch.as_tensor(memo_value,     dtype=torch.float32, device=device).unsqueeze(-1)
            states = torch.as_tensor(memo_state,   dtype=torch.float32, device=device)
            actions= torch.as_tensor(memo_action,  dtype=torch.float32, device=device)

            for batch in batches:
                bs = torch.as_tensor(batch, dtype=torch.long, device=device)

                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(states[bs])
                    old_pi = torch.distributions.Normal(old_mu, old_sigma)
                    # 先 slice 再算 log_prob，并对动作维度求和 -> [B,1]
                    old_logp = old_pi.log_prob(actions[bs]).sum(-1, keepdim=True)

                mu, sigma = self.actor(states[bs])
                pi = torch.distributions.Normal(mu, sigma)
                logp = pi.log_prob(actions[bs]).sum(-1, keepdim=True)

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv[bs]
                surr2 = torch.clamp(ratio, 1 - self.eposilon_clip, 1 + self.eposilon_clip) * adv[bs]
                actor_loss = -torch.min(surr1, surr2).mean()

                returns = adv[bs] + vals[bs]
                values  = self.critic(states[bs])
                critic_loss = torch.nn.functional.mse_loss(values, returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.replay_buffer.clear_memo()


    def save_policy(self):
        torch.save(self.actor.state_dict(), 'ppo_policy_pendulum_v1.para')