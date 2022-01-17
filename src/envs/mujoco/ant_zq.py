import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_to_matrix)

class AntZQ(gym.Wrapper):

    def __init__(self, h=0.4, q=[1., 0., 0., 0.], metric_type="matrix"):
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]
        self.q = np.array(q)
        self.h = h
        self.metric_type = metric_type

    def q_distance(self, s, g, metric_type):
        if metric_type == "angle":
            q_diff = quaternion_multiply(quaternion_invert(s), g)
            q_diff = quaternion_to_angle(q_diff)
        elif metric_type == "matrix":
            rs = quaternion_to_matrix(s)
            rg = quaternion_to_matrix(g)
            q_diff = np.sum((rs - rg)**2)
        return q_diff

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # calculate reward
        state = observation[1:5]
        height = observation[0]
        q_diff = self.q_distance(state, self.q, self.metric_type)
 
        #q_diff = quaternion_metric(state, self.q)
        dh = np.linalg.norm(height - self.h)
        reward = np.exp(-q_diff) 
        info['dq'] = q_diff
        #info['aq'] = aq

        info['dh'] = dh
        return self.state_vector(), reward, done, info

    def reset(self):
        super().reset() 
        qpos = self.init_qpos.copy()
        qpos[3:7] = np.array([0, 0, 1, 0])
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self.state_vector()
        
    #def reset(self):
    #    super().reset()
    #    return self.state_vector()
