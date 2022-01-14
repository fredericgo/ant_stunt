
import gym
import numpy as np
from src.envs.wrappers import ErgodicEnv
import torch
import pathlib
import sys
sys.path.append('./')

import argparse
import imageio
from src.policy import GaussianPolicy
from src.envs.ant import Ant
from src.envs.upsidedown_ant import UpsideDownAnt
from src.envs.wrappers import CustomRewardEnv

parser = argparse.ArgumentParser(description='visualize policy')
parser.add_argument('--policy_weights', type=str, default=None, help='Path to the weights')
parser.add_argument('--output', type=str, default=None, help='output file name')
parser.add_argument('--hidden_features', type=int, default=256, help='NN hidden feature size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--interactive', action='store_true', help='interactive mode')
parser.add_argument('--cuda', action='store_true', help='use cuda')

args = parser.parse_args()

def load_policy(policy, policy_path):
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def select_action(policy, state, evaluate=False):
    state = torch.FloatTensor(state).unsqueeze(0)

    if evaluate is False:
        action, _, _ = policy.sample(state)
    else:
        _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

def render_trajectory(
        env, 
        policy,
        writer,
        traj_len=1000):

    states = []
    actions = []

    state = env.reset()
    done = False
    interactive = False if writer else True
    for t in range(traj_len):
        if interactive:
            env.render()
        else:
            writer.append_data(env.render(mode="rgb_array"))

        states.append(state)
        if done:
            state = env.reset()
            break

        action = policy.predict(state).numpy()    
        actions.append(action)
        state, r, done, _ = env.step(action)
    return np.stack(states), np.array(actions)


def render_policy(
        output,
        seed,
        policy_path,
        use_cuda,
        hidden_sizes=(400, 300),):

    if output:
        output_path = pathlib.Path('./results/') / output

    torch.manual_seed(seed)
    np.random.seed(seed)

    def ant_escape(s, r, d, i):
        _self = i['self']
        l1 = _self.unwrapped.get_body_com('aux_1')[2]
        l2 = _self.unwrapped.get_body_com('aux_2')[2]
        l3 = _self.unwrapped.get_body_com('aux_3')[2]
        l4 = _self.unwrapped.get_body_com('aux_4')[2]
        thresh = 0.8
        if l1 >= thresh and l2 >= thresh and l3 >= thresh and l4 >= thresh:
            return 1, True
        else:
            return 0, False
    env = CustomRewardEnv(UpsideDownAnt(), ant_escape)


    action_dim = env.action_space.shape[0]
    policy = GaussianPolicy(hidden_sizes, env.num_features, action_dim)
    load_policy(policy, policy_path)    

    if output:
        writer = imageio.get_writer(output_path, fps=30) 
    else:
        writer = None

    for _ in range(10):
        render_trajectory(env, policy, writer)

if __name__ == "__main__":
    render_policy(
        output=args.output,
        seed=args.seed,
        policy_path=args.policy_weights,
        use_cuda=args.cuda)
