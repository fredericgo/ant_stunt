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
from envs.mujoco.ant_takeoff import AntTakeoff
from envs.mujoco.ant_walk import AntWalk
from envs.mujoco.ant_runup import AntRunup
from envs.mujoco.ant_jump import AntJump
from envs.mujoco.half_cheetah_runup import HalfCheetahRunup
from envs.mujoco.hopper_runup import HopperRunup

env_names = ['ant', 'ant_walk',  'ant_jump', 'ant_runup', 
             'half_cheetah_runup', 'hopper_runup']

def create_env(env_name, **kwargs):
    """Helper function."""
    assert env_name in env_names

    if env_name == 'ant':
        return Ant()
    elif env_name == 'ant_walk':
        return AntWalk(**kwargs)
    elif env_name == 'ant_jump':
        return AntJump(**kwargs)
    elif env_name == 'ant_runup':
        return AntRunup(**kwargs)
    elif env_name == 'half_cheetah_runup':
        return HalfCheetahRunup(**kwargs)
    elif env_name == 'hopper_runup':
        return HopperRunup(**kwargs)
