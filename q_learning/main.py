import gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode="human")

# 离散化状态空间
obs_size = [20] * len(env.observation_space.high)
win_size = (env.observation_space.high - env.observation_space.low) / obs_size
Q_table_size = obs_size + [env.action_space.n]
Q_table = np.random.uniform(low=-2, high=0, size=Q_table_size)

# 超参数
EPISODES = 5000
DISCOUNT = 0.95
LEARNING_RATE = 0.1
epsilon = 0.5
MIN_EPSILON = 0.01
epsilon_decay = (epsilon - MIN_EPSILON) / EPISODES  # 线性衰减

STATUS_EVERY = 100
epi_status = {'ep': [], 'avg': [], 'max': [], 'min': []}
rewards_status_every = []


def get_qtable_position(state):
    position = (state - env.observation_space.low) / win_size
    return tuple(position.astype(np.int32))


for episode in range(EPISODES):
    # obs 兼容新版 gym
    obs, _ = env.reset()
    # env.render()
    position = get_qtable_position(obs)
    done = False
    epi_reward = 0

    while not done:
        # epsilon-greedy 策略
        if np.random.random() > epsilon:
            action = np.argmax(Q_table[position])
        else:
            action = env.action_space.sample()

        # 兼容新版本 Gym 的 step 返回
        step_result = env.step(action)
        if len(step_result) == 5:  # (obs, reward, terminated, truncated, info)
            new_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # 老版本 Gym
            new_obs, reward, done, info = step_result

        # Q-learning 更新
        new_position = get_qtable_position(new_obs)
        q_current = Q_table[position + (action,)]
        if not done:
            q_future_max = np.max(Q_table[new_position])
            q_new = (1 - LEARNING_RATE) * q_current + LEARNING_RATE * (reward + DISCOUNT * q_future_max)
            Q_table[position + (action,)] = q_new
        else:
            # 到达目标
            if new_obs[0] >= env.goal_position:
                print(f"Reached goal at episode {episode}")
                Q_table[position + (action,)] = 0

        position = new_position
        epi_reward += reward

    # 记录训练状态
    rewards_status_every.append(epi_reward)
    if episode % STATUS_EVERY == 0 and episode > 0:
        avg = np.mean(rewards_status_every)
        epi_status['ep'].append(episode)
        epi_status['avg'].append(avg)
        epi_status['max'].append(max(rewards_status_every))
        epi_status['min'].append(min(rewards_status_every))
        rewards_status_every.clear()
        print(f"Episode: {episode}, Avg Reward: {avg:.2f}, Epsilon: {epsilon:.3f}")
        env.render()

    # 衰减 epsilon
    if epsilon > MIN_EPSILON:
        epsilon -= epsilon_decay

env.close()
