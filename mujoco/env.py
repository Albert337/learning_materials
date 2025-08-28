import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        xml_path = "./xml/robot.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.render_mode = render_mode

        self.viewer = None
        self.offscreen = None  # 离屏渲染用


    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = 1.0 if abs(obs[2]) < 0.2 else 0.0  # reward = upright
        done = False
        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 更新 viewer 内容
            mujoco.viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            if self.offscreen is None:
                self.offscreen = mujoco.Renderer(self.model, height=480, width=640)
            self.offscreen.update_scene(self.data)
            rgb = self.offscreen.render()
            return np.asarray(rgb, dtype=np.uint8)

        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not supported")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.offscreen is not None:
            self.offscreen.close()
            self.offscreen = None
