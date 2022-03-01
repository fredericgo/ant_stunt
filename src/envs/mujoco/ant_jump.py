from tkinter import S
import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_apply)
from common.vector_util import angle_between_2d
from common.bell import bell

class AntJump(gym.Wrapper):

    def __init__(self, 
                 z,
                 initial_state_file,
                 max_episode_steps=100,
                 init_threshold=0.5,
                 threshold=0.1,
                 threshold_inc=0.1):
        """
        The goal of the env is to train a ant that reach target height and pose.
        Z axis is up down.
        T: episode length
        q: orientatoin
        """
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True,
                           terminate_when_unhealthy=True)

        super().__init__(ant_env)

        self.num_features = 27
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.z = z
        self.final_threshold = threshold
        self.threshold = init_threshold
        self.threshold_inc = threshold_inc
        self.t = 0
        self.passed = False
        self.head_landed = False

        self.load_initial_state(initial_state_file)

    def load_initial_state(self, file_name):
        initial_state = np.load(file_name)
        self.init_qpos = np.concatenate([[0., 0.], initial_state[1:14]])
        self.init_qvel = initial_state[14:]

    def update_threshold(self, avg_reward, avg_reward_threshold=0.8):
        """
        call increment threshold when average evaluation reward is higer than 0.5
        """
        if avg_reward < avg_reward_threshold:
            return
        self.threshold -= self.threshold_inc
        self.threshold = np.maximum(self.threshold, self.final_threshold)

    def is_passed(self):
        return self.passed 

    def test_pass(self, dz):
        self.passed = (dz <= self.threshold)

    def test_head_landing(self):
        torso_id = 1
        head_contacts = [x for x in self.sim.data.contact if x.geom1 == torso_id or x.geom2 == torso_id]
        if len(head_contacts) > 0:
            self.head_landed = True

    def reward_func(self, state, action, info={}):       
        reward = 0.0 
        info["dv"] = 0.0

        z = state[2]
        dz = np.linalg.norm(z - self.z)
        if not self.is_passed():
            self.test_pass(dz)

        info['dq'] = dz
        self.test_head_landing()

        if self._elapsed_steps >= self._max_episode_steps:            
            reward = int(self.is_passed())
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
        return observation[:27], reward, done, info

    def reset(self):
        super().reset() 
        self._elapsed_steps = 0
        self.t = 0
        self.passed = False
        self.head_landed = False
        qpos = self.init_qpos.copy()        
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()
        return observation[:27]
        