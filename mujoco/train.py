import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# 这里假设你已经实现了 InvertedPendulumEnv
from env import InvertedPendulumEnv  

# ========== 配置 ==========
TOTAL_TIMESTEPS = 50000     # 总训练步数
EVAL_INTERVAL   = 50        # 每隔多少次迭代可视化
TIMESTEPS_PER_ITER = 1000   # 每次迭代训练步数
SAVE_DIR        = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 创建环境 ==========
def make_env():
    return InvertedPendulumEnv(render_mode=None)

env = DummyVecEnv([make_env])   # SB3 要求VecEnv

# ========== 定义模型 ==========
model = PPO("MlpPolicy", env, verbose=1)

best_reward = -np.inf   # 记录最佳奖励
n_iters = TOTAL_TIMESTEPS // TIMESTEPS_PER_ITER

for i in range(1, n_iters + 1):
    # 训练
    model.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False)

    # 评估
    test_env = InvertedPendulumEnv(render_mode=None)
    obs, _ = test_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        episode_reward += reward
    test_env.close()

    print(f"🔎 Iter {i} -- 评估奖励: {episode_reward:.2f}")

    # 保存最优模型
    if episode_reward > best_reward:
        best_reward = episode_reward
        save_path = os.path.join(SAVE_DIR, "ppo_pendulum_best.zip")
        model.save(save_path)
        print(f"✅ 模型更新: 奖励提升至 {best_reward:.2f}，已保存 {save_path}")

    # 每隔 50 次迭代可视化一次
    if i % EVAL_INTERVAL == 0:
        vis_env = InvertedPendulumEnv(render_mode="human")
        obs, _ = vis_env.reset()
        done = False
        print(f"🎬 可视化 iteration {i}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = vis_env.step(action)
            done = terminated or truncated
            vis_env.render()
        vis_env.close()

# ========== 结束 ==========
env.close()
print("🚀 训练完成！ 最佳奖励:", best_reward)

