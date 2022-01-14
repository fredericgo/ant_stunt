import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box

from common.np_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_to_matrix,
                            quaternion_apply)
from pathlib import Path

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
xml_file = str(Path(__file__).resolve().parent / "assets" / "ant_goal.xml")


class AntGoalXYEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_threshold=0.5, noise=1.0, dense=True):
        self.noise = noise
        self.goal_space = Box(low=-np.inf, high=np.inf, shape=(3,))        
        self.goal = np.array([0.0, 0.0, 0.75], dtype=np.float32)

        self.goal_threshold = goal_threshold
        self.goal_idx = 14

        self.dense = dense

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        state = self.state_vector()
        reward, distance = self.reward_func(state)

        notdone = np.isfinite(state).all() and state[2] >= 0.25 and state[2] <= 1.0
        done = not notdone
        ob = self.state_vector()
        info = {"goal_distance": distance}
        return (ob, reward, done, info,)

    def reward_func(self, state):
        if self.dense:
            return self._dense_reward(state)
        return self._sparse_reward(state)

    def _sparse_reward(self, state):
        # TODO: Deepmimic use q1*q0.conj() -> then calculate the angle 
        goal = state[..., :3]
     
        dis = np.linalg.norm(goal - self.goal)
        reward = 0
        if dis < self.goal_threshold:
            reward = 1
        return reward, dis

    def _dense_reward(self, state):
        # TODO: Deepmimic use q1*q0.conj() -> then calculate the angle 
        goal = state[..., :3]
        dis = np.linalg.norm(goal - self.goal)

        reward = np.exp(-dis)
        return reward, dis

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ]
        )
    
    def set_init_state_and_goal(self, init_state, goal):
        self.goal = goal

        qpos = init_state[:self.model.nq]
        qvel = init_state[self.model.nq:]
        self.set_state(qpos, qvel)
        
    def reset_model(self):
        nq = self.model.nq
        qpos = self.init_qpos.copy()
        qpos[2] = .26
        qpos[:3] += np.random.uniform(-1., 1., 3) * self.noise
        qpos[2] = np.clip(qpos[2], 0.26, 1)

        qvel = self.init_qvel.copy() 
        self.set_state(qpos, qvel)

        # set goal 
        self.model.body_pos[self.goal_idx] = self.goal
        return self.state_vector()
    
    def sample_goal(self):
        """
        Samples a goal state (of type self.state_space.sample()) using 'desired_goal'
        
        """
        x = np.random.uniform(-1., 1., 3) * self.noise
        x[2] = np.clip(x[2], 0.26, 1)
        self.goal = x
        return self.goal

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * .5

    def goal_distance(self, state, goal):
        if self.goal_metric == 'euclidean':
            diff = state[..., :3] - goal
            return np.linalg.norm(diff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)