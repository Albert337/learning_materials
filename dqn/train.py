import gym
import torch 
import torch.nn as nn
import numpy as np
from agent import DQNAgent


env = gym.make("CartPole-v1")

# reset() 在新版 gym 里返回 (obs, info)
s, _ = env.reset()

EPSILON_DECAY = 100000
EPSIOLON_START = 1.0
EPSILON_END = 0.02
TARGET_UPDATE_INTERVAL = 10

n_episodes = 5000
n_time_steps = 1000
REWARD_BUFFER = np.empty(shape=n_episodes)
n_state = len(s)
n_action = env.action_space.n
agent = DQNAgent(n_state, n_action)

for episode_i in range(n_episodes):
    s, _ = env.reset()   # 每个 episode 重置环境
    episode_reward = 0
    
    for step_i in range(n_time_steps):
        epsilon = np.interp(episode_i * n_time_steps + step_i, 
                            [0, EPSILON_DECAY], 
                            [EPSIOLON_START, EPSILON_END])
        random_sample = np.random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)

        # step() 在新版 gym 里返回 5 个值
        s_, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated

        agent.memo.add_memo(s, a, r, s_, done)
        s = s_
        episode_reward += r

        if done:
            REWARD_BUFFER[episode_i] = episode_reward
            break

        if np.mean(REWARD_BUFFER[:episode_i])>=100:
            while True:
                a=agent.online_net.act(s)
                s_, r, terminated, truncated,info = env.step(a)
                env.render()
                done = terminated or truncated
                if done:
                    env.reset()

        
        # 从经验池采样
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        ### compute target
        target_q_value = agent.target_net(batch_s_)
        max_target_q_values = target_q_value.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.gamma * max_target_q_values * (1 - batch_done)

        ### compute q_value
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(q_values, 1, batch_a)

        loss = nn.functional.l1_loss(a_q_values, targets)

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
    
    # 更新 target 网络
    if episode_i % TARGET_UPDATE_INTERVAL == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())


    # ###软更新，每次训练都轻微更新，收敛更平滑。
    # tau = 0.005
    # for target_param, param in zip(agent.target_net.parameters(), agent.online_net.parameters()):
    #     target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


    ## show result
    print("Episode:", episode_i)
    print("Reward:", np.mean(REWARD_BUFFER[:episode_i+1]))
