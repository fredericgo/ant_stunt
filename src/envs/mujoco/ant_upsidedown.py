import numpy as np
import gym

from common.quaternion_util import (quaternion_invert, quaternion_multiply, 
                            quaternion_to_angle,
                            quaternion_to_matrix,
                            quaternion_apply)


class AntUpsidedown(gym.Wrapper):

    def __init__(self):
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # calculate reward
        state = observation[1:5]
        q_diff = quaternion_multiply( 
                    quaternion_invert(state),
                    np.array([1, 0, 0, 0]))
        q_diff = quaternion_to_angle(q_diff)
        reward = np.exp(-q_diff)
        info['dq'] = q_diff

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
