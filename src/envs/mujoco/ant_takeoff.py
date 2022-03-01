from tkinter import S
import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)
from common.vector_util import angle_between_2d
from common.bell import bell

class AntTakeoff(gym.Wrapper):

    def __init__(self, 
                 vy, 
                 angular_velocity,
                 max_episode_steps=100,
                 ctrl_cost_weight=0.5,
                 bonus_weight=100,
                 max_speed=3.0):
        """
        The goal of the env is to train a ant that takes off at the end of an episode.
        Z axis is up down.
        T: episode length
        facing direction: the x axis of the root joint
        forward reference axis: global x-axis
        forward direction: facing direction projected to xy plane
        vz: target facing velocity [0, 3]
        omega0: target angular velocity
        """
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True,
                           terminate_when_unhealthy=False)

        super().__init__(ant_env)

        self.num_features = 1 + self.env.unwrapped._get_obs().shape[0] 
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.forward_ref_dir = np.array([1, 0, 0], dtype=np.float64)
        self.vy = vy
        self.angular_velocity = angular_velocity
        self.max_speed = max_speed

        self._ctrl_cost_weight = ctrl_cost_weight
        self._bonus_weight = bonus_weight
        self.t = 0

    def calc_front_direction(self, state):
        q = state[1:5]
        x_p = quaternion_apply(q, self.forward_ref_dir)
        return angle_between_2d(x_p[:2], self.forward_ref_dir[:2])

    def get_forward_vel(self, state):
        v = state[15:18]
        vf = np.dot(v, self.forward_ref_dir)
        vf /= (np.linalg.norm(self.forward_ref_dir) + 1e-6) 
        return vf

    def bonus_func(self, state, info={}):
        alpha = self.calc_front_direction(state)
        da = alpha**2
        
        v = state[15:18]
        bonus = self._bonus_weight * bell(v[1], self.vy, 1.0)
        omega = state[18:21]
        bonus *= np.exp(-1.0* np.mean(np.abs(omega - self.angular_velocity)))

        info["dv"] = bonus
        return bonus

    def reward_func(self, state, action, info={}):
        vf = self.get_forward_vel(state)
        vf = np.clip(vf, -self.max_speed, self.max_speed)
        info["dv"] = vf
        reward = (vf + self.max_speed) / self.max_speed - 1.0
        ctrl_cost = 0.5 * np.square(action).sum()
       
        reward = reward - ctrl_cost 

        if self._elapsed_steps >= self._max_episode_steps:
            reward = self.bonus_func(state, info)
        return reward

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["truncated"] = not done
            done = True
        
        # calculate reward
        state = self.state_vector()
        reward = self.reward_func(state, action, info)
        self.t += self.dt        
        return np.concatenate([[self.t], observation]), reward, done, info

    def reset(self):
        super().reset() 
        self._elapsed_steps = 0
        self.t = 0
        qpos = self.init_qpos.copy()
        #qpos[3:7] = np.array([0, 0, 1, 0])
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()
        return np.concatenate([[self.t], observation])
        