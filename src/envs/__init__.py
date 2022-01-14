"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

import numpy as np

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')

import gym
from gym.wrappers import TimeLimit

from envs.wrappers import ErgodicEnv

from envs.mujoco.ant import Ant
from envs.mujoco.ant_upsidedown import AntUpsidedown
from envs.mujoco.ant_random import AntRandom
from envs.mujoco.ant_zq import AntZQ

env_names = ['ant', 'ant_upsidedown', 'ant_random', 'ant_zq']

def create_env(env_name, **kwargs):
    """Helper function."""
    assert env_name in env_names

    if env_name == 'ant':
        return ErgodicEnv(Ant())
    elif env_name == 'ant_upsidedown':
        return ErgodicEnv(AntUpsidedown())
    elif env_name == 'ant_random':
        return ErgodicEnv(AntRandom())
    elif env_name == 'ant_zq':
        return ErgodicEnv(AntZQ(**kwargs))
    

from envs.goal.ant_goal import AntGoalEnv
from envs.goal.ant_goal_xy import AntGoalXYEnv

goal_env_names = ['ant_goal', 'ant_goal_xy']

def create_goal_env(env_name, **kwargs):
    assert env_name in goal_env_names

    if env_name == 'ant_goal':
        return ErgodicEnv(AntGoalEnv(**kwargs))
    elif env_name == 'ant_goal_xy':
        return ErgodicEnv(AntGoalXYEnv(**kwargs))
