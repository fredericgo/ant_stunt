import numpy as np
import gym


class Ant(gym.Wrapper):

    def __init__(self):
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0] -2

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.state_vector()[2:], reward, done, info

    def reset(self):
        super().reset()
        return self.state_vector()[2:]
