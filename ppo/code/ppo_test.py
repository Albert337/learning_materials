# test_ppo.py
import gym
import torch
import numpy as np
from ppo_agent import PPOAgent, Actor, Critic, device

# 场景要和训练时一致
Scene = "Pendulum-v1"
env = gym.make(Scene, render_mode="human")   # 如果用 jupyter, 可以改 render_mode="rgb_array"

# 状态、动作维度
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# 创建 agent
agent = PPOAgent(STATE_DIM, ACTION_DIM, batch_size=64)

# 加载模型参数（确保文件路径和训练时一致）
actor_path = "ppo_policy_pendulum_v1.pth"
# critic_path = "ppo_value_pendulum_v1.pth"

agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
# agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
agent.actor.eval()
# agent.critic.eval()

# 测试 N 条 episode
NUM_TEST_EPISODES = 5
NUM_STEPS = 200

for ep in range(NUM_TEST_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(NUM_STEPS):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, sigma = agent.actor(state_tensor)
            dist = torch.distributions.Normal(mu, sigma)
            action = mu   # 这里用平均值动作，而不是采样，保证更稳定
            action = action.cpu().numpy().astype(np.float32).reshape(-1)

        # 执行动作
        state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if done or truncated:
            break

    print(f"Test Episode {ep+1}: reward = {round(episode_reward, 2)}")

env.close()
