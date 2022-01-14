import argparse
import datetime
import envs

import numpy as np
import itertools
import torch
import pathlib
import imageio

from common.chamfer import chamfer_distance

from skill_learning.networks import GaussianPolicy

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--traj_len', type=int, default=500, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_epochs', type=int, default=200, 
                    help='num epochs')
parser.add_argument('--num_skills', type=int, default=10, 
                    help='num epochs')                
parser.add_argument('--model_dir', type=str, default=None, 
                    help="model path")
parser.add_argument('--video_file_name', type=str, default=None, 
                    help='video file name')
args = parser.parse_args()


def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def sample_skill():
        return torch.randint(args.num_skills, ())

def render_trajectory(policy, env, skill):

    state = env.reset()
    skill = torch.tensor(skill)
    skill = skill.unsqueeze(0)

    done = False
    traj = []
    for t in range(args.traj_len):

        if done:
            state = env.reset()

        traj.append(state)

        state = torch.FloatTensor(state).unsqueeze(0)
   
        _, _, action = policy.sample(state, skill)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        state, r, done, _ = env.step(action)
    
    return np.stack(traj)

def main():
    # Environment
    env = envs.create_env(args.env_name)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    policy = GaussianPolicy(env.num_features, env.action_space.shape[0], args.num_skills, args.hidden_size, env.action_space)
    load_policy(policy, args.model_dir)
    
    trajectories = []
    for i in range(3):
        traj = render_trajectory(policy, env, i)
        trajectories.append(traj)

    res = dict()
    for i in range(3):
        for j in range(3):
            if j > i:
                res[(i, j)] = chamfer_distance(trajectories[i], trajectories[j])
    print(res)
    env.close()

if __name__ == "__main__":
    main()