import numpy as np
import gym

class AntRandom(gym.Wrapper):

    def __init__(self, noise_level=0.5):
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]
        self.noise_level = noise_level

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.state_vector(), reward, done, info

    def reset(self):
        super().reset() 
        qpos = self.init_qpos.copy()

        v = np.array([1, 0, 0, 0]) + np.random.randn(4) * self.noise_level
        v = v / np.linalg.norm(v)
        qpos[3:7] = v
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self.state_vector()
        