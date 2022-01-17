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
from envs.mujoco.ant_takeoff import AntTakeoff

env_names = ['ant', 'ant_upsidedown', 'ant_takeoff']

def create_env(env_name, **kwargs):
    """Helper function."""
    assert env_name in env_names

    if env_name == 'ant':
        return ErgodicEnv(Ant())
    elif env_name == 'ant_upsidedown':
        return ErgodicEnv(AntUpsidedown())
    elif env_name == 'ant_takeoff':
        return AntTakeoff(**kwargs)

