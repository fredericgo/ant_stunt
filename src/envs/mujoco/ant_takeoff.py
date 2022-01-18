import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)
from common.vector_util import angle_between_2d

class AntTakeoff(gym.Wrapper):

    def __init__(self, 
                 v0, 
                 max_episode_steps=100,
                 ctrl_cost_weight=0.5,
                 bonus_weight=10):
        """
        The goal of the env is to train a ant that takes off at the end of an episode.
        Z axis is up down.
        T: episode length
        facing direction: the x axis of the root joint
        forward reference axis: global x-axis
        forward direction: facing direction projected to xy plane
        v0: target velocity [-6, 6]
        omega0: target angular velocity
        """
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True,
                           terminate_when_unhealthy=False)

        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.forward_ref_dir = np.array([1, 0, 0], dtype=np.float64)
        self.v0 = v0
        self._ctrl_cost_weight = ctrl_cost_weight
        self._bonus_weight = bonus_weight

    def calc_front_direction(self, state):
        q = state[3:7]
        x_p = quaternion_apply(q, self.forward_ref_dir)
        return angle_between_2d(x_p[:2], self.forward_ref_dir[:2])

    def get_linear_vel(self, state):
        return state[15:18]

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def reward_func(self, state, info={}):
        #if self._elapsed_steps < self._max_episode_steps:
        #    v = self.get_linear_vel(state)
        #    dv = np.linalg.norm(v[:2] - self.v0[:2], ord=1)
        #    info["dv"] = dv
        #    return np.exp(-dv)

        alpha = self.calc_front_direction(state)
        da = alpha**2
        
        v = self.get_linear_vel(state)
        dv = np.linalg.norm(v - self.v0, ord=1)
        
        reward = self._bonus_weight * np.exp(-dv)
        info["dv"] = dv
        return reward

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["truncated"] = not done
            done = True
        
        # calculate reward
        state = self.state_vector()
        bonus = self.reward_func(state, info)
        cost = self.control_cost(action)

        reward = bonus -  cost
        
        return self.state_vector(), reward, done, info

    def reset(self):
        super().reset() 
        self._elapsed_steps = 0
        qpos = self.init_qpos.copy()
        #qpos[3:7] = np.array([0, 0, 1, 0])
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self.state_vector()
        