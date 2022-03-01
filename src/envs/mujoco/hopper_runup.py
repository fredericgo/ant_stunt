from tkinter import S
import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)

from common.bell import bell


class HopperRunup(gym.Wrapper):

    def __init__(self, 
                 velocity, 
                 angular_velocity,
                 max_episode_steps=100,
                 bonus_weight=100):
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
        ant_env = gym.make('Hopper-v2')

        super().__init__(ant_env)

        self.num_features = 11 
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self._bonus_weight = bonus_weight

        self.t = 0

    def bonus_func(self, state, info={}):
        """
        dof description:
        0: x
        1: z
        2: y (rotation)
        3-6: joints
        """
        pose_w     = 1.

        theta = state[3:6]
        theta0 = self.init_qpos[3:]

        joint_err = (theta - theta0)**2   
        joint_err = np.sum(joint_err, axis=-1)

        bonus0 = pose_w * np.exp(-1 * joint_err)
        
        v = state[6:8] # x z (linear)
        bonus = self._bonus_weight * np.exp(-1.0* np.mean(np.abs(v - self.velocity)))
        omega = state[8] # y (angular)
        bonus *= np.exp(-1.0* np.mean(np.abs(omega - self.angular_velocity)))

        info["dv"] = bonus
        return bonus0 + bonus 

    def reward_func(self, state, action, info={}):  
        reward = 0.
        if self._elapsed_steps >= self._max_episode_steps:
            reward += self.bonus_func(state, info)
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
        return observation, reward, done, info

    def reset(self):
        observation = super().reset() 
        self._elapsed_steps = 0
        self.t = 0
        return observation
        