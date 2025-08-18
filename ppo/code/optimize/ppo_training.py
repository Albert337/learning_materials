import gym
import numpy as np
import torch
from ppo_agent import PPOAgent  # 这里引用上面写的 PPOAgent 代码

# -----------------------------
# 环境 & Agent 初始化
# -----------------------------
env = gym.make("Pendulum-v1")  # 适合测试连续动作 PPO
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = env.action_space.low[0]
action_high = env.action_space.high[0]

agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    batch_size=64,
    gamma=0.97,
    lamada=0.95,
    epochs=10,
    clip_eps=0.2,
    actor_lr=1e-4,
    critic_lr=1e-3,
    entropy_coef=1e-2,
    action_low=action_low,
    action_high=action_high
)

# -----------------------------
# 训练超参数
# -----------------------------
MAX_EPISODES = 1000
ROLLOUT_STEPS = 2048   # 每次收集多少步再更新
UPDATE_INTERVAL = ROLLOUT_STEPS
best_reward = -np.inf

# -----------------------------
# 训练主循环
# -----------------------------
total_steps = 0
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    while True:
        # 与环境交互
        action, value = agent.get_action(state)   # action: np, value: np
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 存储轨迹 (保持顺序: state, action, value, done, reward)
        agent.replay_buffer.add_memo(state, action, value, done, reward)

        state = next_state
        episode_reward += reward
        total_steps += 1

        # 如果收集满一个 rollout，就更新 PPO
        if total_steps % UPDATE_INTERVAL == 0:
            agent.update()

        if done:
            break

    print(f"Episode {episode} | Reward: {episode_reward:.2f}")

    # 保存最优模型
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save_policy("ppo_pendulum_best.pth")
        print(f"✅ Saved best model with reward {best_reward:.2f}")

env.close()
