import torch
from torch import nn
import numpy as np

# -----------------------
# Utils
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def atanh(x):
    # clamp 避免数值溢出
    x = torch.clamp(x, -1.0 + 1e-6, 1.0 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# -----------------------
# Actor / Critic
# -----------------------
class Actor(nn.Module):
    """
    Squashed-Gaussian Policy with Tanh + action scaling to [low, high].
    提供 log_prob 修正项，适合连续动作 PPO。
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_low=-2.0, action_high=2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # 动作缩放参数
        self.register_buffer("act_low", torch.as_tensor(np.full(action_dim, action_low, dtype=np.float32)))
        self.register_buffer("act_high", torch.as_tensor(np.full(action_dim, action_high, dtype=np.float32)))
        self.register_buffer("act_scale", (self.act_high - self.act_low) / 2.0)
        self.register_buffer("act_center", (self.act_high + self.act_low) / 2.0)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu(h)
        log_std = self.log_std(h)
        # 限制 std 的范围，训练更稳定
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return mu, std

    def _squash(self, unsquashed):
        # tanh -> [-1,1] 再线性缩放到 [low, high]
        squashed = torch.tanh(unsquashed)
        return self.act_center + squashed * self.act_scale, squashed

    def sample(self, x):
        """
        采样动作（带 reparameterization），并返回 squashed 后的对数概率（含修正项）。
        """
        mu, std = self(x)
        normal = torch.distributions.Normal(mu, std)
        eps = normal.rsample()  # reparameterization
        action, squashed = self._squash(eps)

        # log_prob 修正：log_prob(eps) - sum log |det J_tanh|
        # tanh 的雅可比：1 - tanh(z)^2
        base_log_prob = normal.log_prob(eps)  # [B, A]
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)  # [B, A]
        log_prob = base_log_prob - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # [B, 1]
        entropy = normal.entropy().sum(dim=-1, keepdim=True)  # 近似，用基础 Normal 的熵

        return action, log_prob, entropy

    def deterministic(self, x):
        mu, _ = self(x)
        action, _ = self._squash(mu)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------
# Replay Buffer
# -----------------------
class ReplayMemory:
    def __init__(self, batch_size):
        self.state_cap = []
        self.action_cap = []
        self.value_cap = []
        self.done_cap = []
        self.reward_cap = []
        self.batch_size = batch_size

    def add_memo(self, state, action, value, done, reward):
        # 统一 float32
        self.state_cap.append(np.asarray(state, dtype=np.float32))
        self.action_cap.append(np.asarray(action, dtype=np.float32))
        self.value_cap.append(np.asarray(value, dtype=np.float32))
        self.done_cap.append(np.asarray(done, dtype=np.float32))
        self.reward_cap.append(np.asarray(reward, dtype=np.float32))

    def sample(self):
        num = len(self.state_cap)
        # 全量打乱 -> 再按 batch 切
        idx = np.arange(num, dtype=np.int32)
        np.random.shuffle(idx)
        batches = [idx[i:i+self.batch_size] for i in range(0, num, self.batch_size)]

        return (
            np.array(self.state_cap, dtype=np.float32),
            np.array(self.action_cap, dtype=np.float32),
            np.array(self.reward_cap, dtype=np.float32),
            np.array(self.value_cap, dtype=np.float32).squeeze(-1),  # [T] (兼容你之前的 shape)
            np.array(self.done_cap, dtype=np.float32),
            batches
        )

    def clear_memo(self):
        self.state_cap.clear()
        self.action_cap.clear()
        self.value_cap.clear()
        self.done_cap.clear()
        self.reward_cap.clear()


# -----------------------
# PPO Agent
# -----------------------
class PPOAgent:
    def __init__(
        self,
        state_dim, action_dim, batch_size,
        gamma=0.97, lamada=0.95,
        epochs=10, clip_eps=0.2,
        actor_lr=1e-4, critic_lr=1e-3,
        entropy_coef=1e-2,
        action_low=-2.0, action_high=2.0
    ):
        super().__init__()
        self.gamma = gamma
        self.lamada = lamada
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        self.actor = Actor(state_dim, action_dim, action_low=action_low, action_high=action_high).to(device)
        self.old_actor = Actor(state_dim, action_dim, action_low=action_low, action_high=action_high).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayMemory(batch_size)

    @torch.no_grad()
    def get_action(self, state):
        """
        与环境交互使用：返回 (action_np, value_np)
        action shape: (action_dim,), dtype float32
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, S]
        a, _, _ = self.actor.sample(s)  # [1, A]
        v = self.critic(s)              # [1, 1]
        return a.squeeze(0).cpu().numpy().astype(np.float32), v.squeeze(0).cpu().numpy().astype(np.float32)

    def _compute_gae(self, rewards, values, dones):
        """
        rewards: [T], values: [T] (V_t), dones: [T]
        返回 advantages[T], returns[T]
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        # 末尾 bootstrap 值设为 0（或可传入 V_{T}）
        next_value = 0.0
        next_nonterminal = 1.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lamada * next_nonterminal * gae
            adv[t] = gae
            next_value = values[t]
            next_nonterminal = 1.0 - dones[t]

        returns = adv + values
        return adv, returns

    def update(self):
        # 同步 old_policy
        self.old_actor.load_state_dict(self.actor.state_dict())

        # 取出一整个窗口的轨迹
        memo_state, memo_action, memo_reward, memo_value, memo_done, batches = self.replay_buffer.sample()
        T = len(memo_reward)

        # ---------- GAE ----------
        advantages, returns = self._compute_gae(memo_reward, memo_value, memo_done)

        # 标准化 advantage
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ---------- to Tensor ----------
        states  = torch.as_tensor(memo_state,  dtype=torch.float32, device=device)       # [T, S]
        actions = torch.as_tensor(memo_action, dtype=torch.float32, device=device)       # [T, A]
        adv_t   = torch.as_tensor(advantages,  dtype=torch.float32, device=device).unsqueeze(-1)  # [T, 1]
        ret_t   = torch.as_tensor(returns,     dtype=torch.float32, device=device).unsqueeze(-1)  # [T, 1]

        for _ in range(self.epochs):
            for batch in batches:
                b = torch.as_tensor(batch, dtype=torch.long, device=device)

                # old log_prob
                with torch.no_grad():
                    old_mu, old_std = self.old_actor(states[b])
                    old_normal = torch.distributions.Normal(old_mu, old_std)

                    # 将已 squashed 的动作反变换到 unsquashed 空间：
                    # action_squashed in [low, high] -> [-1,1] -> atanh -> unsquashed
                    scale = (self.old_actor.act_high - self.old_actor.act_low) / 2.0
                    center = (self.old_actor.act_high + self.old_actor.act_low) / 2.0
                    a_unit = (actions[b] - center) / (scale + 1e-6)          # [-1,1]
                    unsquashed = atanh(a_unit)                               # R

                    base_log_prob = old_normal.log_prob(unsquashed)          # [B, A]
                    correction = torch.log(1.0 - torch.tanh(unsquashed).pow(2) + 1e-6)  # [B, A]
                    old_logp = (base_log_prob - correction).sum(dim=-1, keepdim=True)   # [B, 1]

                # current log_prob
                mu, std = self.actor(states[b])
                normal = torch.distributions.Normal(mu, std)

                # 复用上面的 unsquashed（按当前策略不严谨，但 PPO 用 ratio 仍然可行；
                # 若要更严格，可重新 unsquash 用当前 scale/center，一般两者相同）
                base_log_prob = normal.log_prob(unsquashed)                  # [B, A]
                correction = torch.log(1.0 - torch.tanh(unsquashed).pow(2) + 1e-6)
                logp = (base_log_prob - correction).sum(dim=-1, keepdim=True)  # [B, 1]

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                # 近似 entropy（来自基础 Normal）
                entropy = normal.entropy().sum(dim=-1, keepdim=True).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                # Critic：用 returns 的 detached 版本
                values = self.critic(states[b])
                critic_loss = nn.functional.mse_loss(values, ret_t[b].detach())

                # step
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()

        # 清空这一轮的轨迹
        self.replay_buffer.clear_memo()

    def save_policy(self, path='ppo_policy_pendulum_v1.pth'):
        torch.save(self.actor.state_dict(), path)
