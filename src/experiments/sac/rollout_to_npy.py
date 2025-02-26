import argparse
import datetime
import envs

import numpy as np
import itertools
import torch
import pathlib
import imageio

from PIL import Image, ImageFont, ImageDraw 

from sac.model import GaussianPolicy

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_walk",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--traj_len', type=int, default=100, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_epochs', type=int, default=1, 
                    help='num epochs')
parser.add_argument('--model_dir', type=str, default=None, 
                    help="model path")
parser.add_argument('--output_file', type=str, default=None, 
                    help='video file name')
args = parser.parse_args()


def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def rollout_trajectory(policy, env):

    state = env.reset()
    done = False
    
    states = []
    states.append(state)
    for t in range(args.traj_len):
        if done:
            state = env.reset()

        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, action = policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        state, r, done, _ = env.step(action)
        states.append(state)
    return np.array(states)


def main():
    # Environment
    env = envs.create_env(
            args.env_name, 
            max_episode_steps=args.traj_len)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = GaussianPolicy(env.num_features, env.action_space.shape[0], args.hidden_size, env.action_space)
    load_policy(policy, args.model_dir)

    traj = rollout_trajectory(policy, env)
    traj = traj[:,:28]

    with open(args.output_file, 'wb') as f:
        np.save(f, traj)
    env.close()

if __name__ == "__main__":
    main()