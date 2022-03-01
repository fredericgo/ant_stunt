from tkinter import S
import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)

from common.bell import bell
from common.geometry import SkeletonGeometry

JOINT_WEIGHTS = np.array([1., .5, .3, .5, .3, .5, .3, .5, .3], dtype=np.float32)


class AntRunup(gym.Wrapper):

    def __init__(self, 
                 velocity, 
                 angular_velocity,
                 motion_file='data/ant_walk_traj.npy',
                 max_episode_steps=100,
                 bonus_weight=10):
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

        self.num_features = 28#self.env.unwrapped._get_obs().shape[0] 
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self._bonus_weight = bonus_weight

        self.motion = np.load(motion_file)
        
        self.skeleton = SkeletonGeometry(self)

        self._heading_vec = np.array([1., 0., 0.])
        self.t = 0

    def bonus_func(self, state, info={}):
        v = state[15:18]
        bonus = self._bonus_weight * np.exp(-1.0* np.mean(np.abs(v - self.velocity)))
        omega = state[18:21]
        bonus *= np.exp(-1.0* np.mean(np.abs(omega - self.angular_velocity)))

        info["dv"] = bonus
        return bonus

    def reward_func(self, state, action, reference_state, info={}):  
        pose_w     = 0.2
        vel_w      = 0.4
        root_w     = 0.4

        root_pos_err = np.sum((state[...,2] - reference_state[...,1])**2)

        q = state[3:7]
        q_ref = reference_state[2:6]

        root_rot_err = quaternion_multiply(q, quaternion_invert(q_ref))
        root_rot_err = quaternion_to_angle(root_rot_err) 

        v_root = state[15:18]
        v_root0 = reference_state[14:17]
        root_vel_err = np.sum((v_root - v_root0)**2)

        v_angvel = state[18:21]
        v_angvel0 = reference_state[17:20]
        root_omg_err = np.sum((v_angvel - v_angvel0)**2)

        root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_omg_err

        theta = state[7:15]
        theta0 = reference_state[6:14]

        joint_diff = (theta - theta0)**2   
        joint_diff *= JOINT_WEIGHTS[1:]
        joint_diff = np.sum(joint_diff, axis=-1)

        v = state[21:]
        v0 = reference_state[20:]
        vel_err = np.sum((v - v0)**2)

        reward = root_w * np.exp(-1 * root_err)
        reward += pose_w * np.exp(-1 * joint_diff)
        if self._elapsed_steps >= self._max_episode_steps:
            vel_w = 0.0
        reward += vel_w * np.exp(-0.1 * vel_err)

        info["dv"] = root_err

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
        ref_state = self.motion[self._elapsed_steps]
        reward = self.reward_func(state, action, ref_state, info)
        self.t += self.dt        
        return np.concatenate([[self.t], observation[:27]]), reward, done, info

    def reset(self):
        super().reset() 
        self._elapsed_steps = 0
        self.t = 0
        qpos = self.init_qpos.copy()
        #qpos[3:7] = np.array([0, 0, 1, 0])
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()
        return np.concatenate([[self.t], observation[:27]])
        