
import gym
import numpy as np
import torch
import pathlib
import sys
sys.path.append('./')

import imageio
from rl.model import GaussianPolicy

output_dir = '/tmp', 
env_name = 'ant_rot_pts'

#model_dir = 'runs/2021-09-30_14-14-06_SAC_ant_nrg_Gaussian'
#model_dir = 'runs/2021-09-30_14-14-32_SAC_ant_rotnrg_Gaussian'
model_dir = 'runs/2021-10-06_14-41-18_SAC_ant_rot_pts_Gaussian'

video_file = 'ant_rot.mp4'
hidden_size = 256
gpu = True
interactive = False if video_file else True
seed = 0

# Envs
import envs
# Algo
import pathlib

if not gpu:
    print('Not using GPU. Will be slow.')

torch.manual_seed(seed)
np.random.seed(seed)

env = envs.create_env(env_name)

action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
policy = GaussianPolicy(obs_dim, action_dim, hidden_size, env.action_space)


max_path_length = 50


def load_policy(model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def select_action(policy, state, evaluate=False):
    state = torch.FloatTensor(state).unsqueeze(0)

    if evaluate is False:
        action, _, _ = policy.sample(state)
    else:
        _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

def sample_trajectory(writer, greedy=False, noise=0):

    states = []
    actions = []

    state = env.reset()
    done = False
    for t in range(max_path_length):
        if interactive:
            env.render()
        else:
            writer.append_data(env.render(mode="rgb_array"))
        states.append(state)
        if done:
            state = env.reset()

        horizon = np.arange(max_path_length) >= (max_path_length - 1 - t) # Temperature encoding of horizon horizon[None],
        action = select_action(policy, state)
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        actions.append(action)
        state, r, done, _ = env.step(action)
    #env.update_time(t+1)
    return np.stack(states), np.array(actions), t+1

load_policy(model_dir)

if video_file:
    writer = imageio.get_writer(video_file, fps=30) 
else:
    writer = None

total_timesteps = 0
for _ in range(20):
    sample_trajectory(writer)


#sample_init(noise=1, render=True)

if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['ant'], #['ant'],
        'gpu': [True],
    }
    pass
