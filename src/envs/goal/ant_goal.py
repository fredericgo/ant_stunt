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


class AntGoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_threshold=0.5, initial_noise=0.5, 
                       noise=1.0, dense=True, goal_metric="angle"):
        self.initial_noise = initial_noise
        self.noise = noise
        self.goal_space = Box(low=-np.inf, high=np.inf, shape=(5,))        
        self.goal = np.array([0.75, 1, 0, 0, 0], dtype=np.float32)

        self.goal_threshold = goal_threshold
        self.goal_idx = 14

        self.dense = dense
        self.goal_metric = goal_metric

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        state = self.state_vector()
        reward, distance = self.reward_func(state, self.goal)

        notdone = np.isfinite(state).all() and state[2] >= 0.25 and state[2] <= 1.0
        done = not notdone
        ob = self.state_vector()
        info = {"goal_distance": distance}
        return (ob, reward, done, info,)

    def reward_func(self, state, goal):
        if self.dense:
            return self.dense_reward(state, goal)
        return self.sparse_reward(state, goal)

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

        xy = init_state[:2]
        self.set_goal(xy, goal)

    def set_goal(self, xy, goal):
        # set goal 
        self.model.body_pos[self.goal_idx][:2] = xy
        self.model.body_pos[self.goal_idx][2] = goal[0]
        self.model.body_quat[self.goal_idx] = goal[1:]
        
    def reset_model(self):
        nq = self.model.nq
        qpos = self.init_qpos.copy()
        v = np.array([1, 0, 0, 0]) + np.random.randn(4) * self.initial_noise
        v = v / np.linalg.norm(v)
        qpos[3:7] = v
        qvel = self.init_qvel.copy() 
        self.set_state(qpos, qvel)

        # set goal 
        xy = qpos[:2]
        self.set_goal(xy, self.goal)
        return self.state_vector()
    
    def sample_goal(self):
        """
        Samples a goal state (of type self.state_space.sample()) using 'desired_goal'
        
        """
        qpos = self.init_qpos.copy()[:7] 
        q = np.random.randn(4) * self.noise
        q = q / np.linalg.norm(q)

        h = qpos[2] + np.random.randn(1) * self.noise
        h = np.maximum(0.26, h)
        self.goal = np.concatenate([h, q])
        return self.goal

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * .5

    def dense_reward(self, state, goal):
        diff = self.goal_distance(state, goal)
        reward = np.exp(-diff)
        return reward, diff

    def sparse_reward(self, state, goal):
        # TODO: Deepmimic use q1*q0.conj() -> then calculate the angle 
        diff = self.goal_distance(state, goal)
        reward = 0
        if diff < self.goal_threshold:
            reward = 1
        return reward, diff

    def goal_distance(self, state, goal):
        if self.goal_metric == 'angle':
            q_diff = quaternion_multiply( 
                    quaternion_invert(state[..., 3:7]),
                    goal[..., 1:])
            q_diff = quaternion_to_angle(q_diff)
            return q_diff
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)