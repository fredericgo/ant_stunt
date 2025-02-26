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
parser.add_argument('--env-name', default="ant_runup",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--v', default=np.array([3.0,  0.00, 0.00]),
                    help='take off velocity')
parser.add_argument('--w', default=np.array([0.00, 0.00, -3.00]),
                    help='take off anguluar velocity')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--traj_len', type=int, default=50, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_epochs', type=int, default=2, 
                    help='num epochs')
parser.add_argument('--model_dir', type=str, default=None, 
                    help="model path")

args = parser.parse_args()


def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def render_trajectory(policy, env):

    state = env.reset()
    done = False
    
    for t in range(args.traj_len):
        if done:
            state = env.reset()

        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, action = policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        state, r, done, info = env.step(action)
        #print(state.shape)
        print(t, r, state[14:20], info)
    return state


def main():
    # Environment
    env = envs.create_env(
            args.env_name, 
            velocity=args.v,
            angular_velocity=args.w,
            max_episode_steps=args.traj_len)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = GaussianPolicy(env.num_features, env.action_space.shape[0], args.hidden_size, env.action_space)
    load_policy(policy, args.model_dir)

    state = render_trajectory(policy, env)

    p = pathlib.Path(args.model_dir)
    file_name = p / 'laststate.npy'
    np.save(file_name, state)
    env.close()

if __name__ == "__main__":
    main()