import numpy as np
import gym
from gym.spaces import Box


class AntGoalEnv(gym.Wrapper):

    def __init__(self, noise_level=0.5, goal_threshold=0.2):
        ant_env = gym.make('Ant-v3',
                           exclude_current_positions_from_observation=True)
        super().__init__(ant_env)

        self.num_features = self.state_vector().shape[0]
        self.noise_level = noise_level

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.goal_space = Box(low=-np.inf, high=np.inf, shape=(5,))        
        num_obs = self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_obs,))     
        self.noise = noise_level
        self.goal_threshold = goal_threshold

    def step(self, action):
        observation, reward, done, info = super().step(action)
        state = self.state_vector()
        reward, distance = self._reward_function(state)
        info['goal_distance'] = distance
        return state, reward, done, info

    def reset(self):
        super().reset() 
        qpos = self.init_qpos.copy()

        v = np.array([1, 0, 0, 0]) + np.random.randn(4) * self.noise_level
        v = v / np.linalg.norm(v)
        qpos[3:7] = v
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)
        return self.state_vector()

    def _reward_function(self, state):
        # TODO: Deepmimic use q1*q0.conj() -> then calculate the angle 
        goal = state[..., 2:7]
        distance = np.linalg.norm(self.goal - goal, ord=2)
        reward = 0
        if distance < self.goal_threshold:
            reward = 1
        return reward, distance 

    def sample_goal(self):
        """
        Samples a goal state (of type self.state_space.sample()) using 'desired_goal'
        
        """
        qpos = self.env.init_qpos.copy()[:7] 
    
        v = qpos[2:] + np.random.randn(5) * self.noise
        v = v / np.linalg.norm(v)
        qvel = self.env.init_qvel.copy()
        self.goal = v
        return self.goal