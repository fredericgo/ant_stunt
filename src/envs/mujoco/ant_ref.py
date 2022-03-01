from tkinter import S
import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)
from common.vector_util import angle_between_2d
from common.bell import bell
from common.geometry import SkeletonGeometry

JOINT_WEIGHTS = np.array([1., .5, .3, .5, .3, .5, .3, .5, .3], dtype=np.float32)


class AntJumpFixedPose(gym.Wrapper):

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
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True,
                           terminate_when_unhealthy=False)

        super().__init__(ant_env)

        self.num_features = 1 + self.env.unwrapped._get_obs().shape[0] 
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self._bonus_weight = bonus_weight
        self.skeleton = SkeletonGeometry(self)
        
        self.ref_pose = np.array([1., 0., 0., 0., 0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])

        self.t = 0

    def bonus_func(self, state, info={}):
        v = state[15:18]
        bonus = self._bonus_weight * np.exp(-1.0* np.mean(np.abs(v - self.velocity)))
        omega = state[18:21]
        bonus *= np.exp(-1.0* np.mean(np.abs(omega - self.angular_velocity)))

        info["dv"] = bonus
        return bonus

    def reward_func(self, state, action, info={}):      
        q = state[2:6]
        q0 = self.ref_pose[:4]
        q_diff = quaternion_multiply(q, quaternion_invert(q0))
        q_diff = quaternion_to_angle(q_diff) 

        theta = state[6:14]
        theta0 = self.ref_pose[4:]
        joint_diff = (theta - theta0)**2   
        joint_diff *= JOINT_WEIGHTS[1:]
        joint_diff = np.sum(joint_diff, axis=-1)

        reward = np.exp(-1 * q_diff)
        reward *= np.exp(-1 * joint_diff)
        info["dv"] = q_diff

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
        