import argparse
import datetime
import envs

import numpy as np
import itertools
import pathlib
import imageio
import torch
from torch.utils.tensorboard import SummaryWriter
from goal_conditioned.episodic_replay_buffer import EpisodicReplayBuffer
from goal_conditioned.networks import GoalGaussianPolicy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_goal",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--horizon', type=int, default=5, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--traj_len', type=int, default=50, 
                    help='checkpoint training model every # steps')
parser.add_argument('--num_episodes', type=int, default=5, 
                    help='# of epochs')  
parser.add_argument('--replay_size', type=int, default=1000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--hidden_size', type=int, nargs="+", default=[400, 300], metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--state_filter', type=list, default=[2, 3, 4, 5, 6],
                    help='state filter')
parser.add_argument('--model_dir', type=str, default=None, 
                    help="model path")
parser.add_argument('--name', type=str, default="goal_conditioned", 
                    help='video file name')
parser.add_argument('--expert_file', type=str, default=None, 
                    help='expert replay buffer')

args = parser.parse_args()



def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def render_trajectory(policy, env, buffer, writer):
    if buffer:
        state, action, goal, _, _, _ = buffer.sample(1, horizon=args.horizon)
        state = state[0]
        goal  = goal[0, args.state_filter]
        env.set_init_state_and_goal(state, goal)
    else:
        state = env.reset()
        goal = env.sample_goal()

    rewards = []
    distances = []
    done = False
    for t in range(args.traj_len):

        writer.append_data(env.render(mode="rgb_array"))
        if done:
            state = env.reset()

        state = torch.FloatTensor(state)
        goal = torch.FloatTensor(goal)

        _, _, action = policy.sample(state[None], goal[None])
        action = action.detach().cpu().numpy()[0]
        
        state, r, done, info = env.step(action)
        rewards.append(r)
        distances.append(info["goal_distance"])
    return rewards, distances
 

def main():
    # Environment
    env = envs.create_goal_env(args.env_name)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    expert_memory = None
    if args.expert_file:
        expert_memory = EpisodicReplayBuffer(args.replay_size, args.seed)
        expert_memory.load(args.expert_file)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.goal_space.shape[0]
    policy = GoalGaussianPolicy(state_dim, action_dim, goal_dim, args.hidden_size, env.action_space)

    load_policy(policy, args.model_dir)

    datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'results/{args.name}/{datetime_st}_{args.env_name}'
    pathlib.Path(log_dir).mkdir(parents=True)
    video_file_name = pathlib.Path(log_dir) / "render.mp4"
    plot_file_name = pathlib.Path(log_dir) / "reward.png"

    writer = imageio.get_writer(video_file_name, fps=30) 
    
    rewards = []
    distances = []
    for _ in range(args.num_episodes):
        r, d = render_trajectory(policy, env, expert_memory, writer)
        rewards.append(r)
        distances.append(d)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    time = np.arange(args.traj_len)
    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)

    axes[0].set_title("reward by step")
 
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Reward")

    axes[0].fill_between(
        time,
        mean - std,
        mean + std,
        alpha=0.1,
        color="r",
    )

    axes[0].plot(
        time, mean, "o-", color="r", label=f"w={args.horizon}, h={args.traj_len}"
    )

    mean = np.mean(distances, axis=0)
    std = np.std(distances, axis=0)

    axes[1].set_title("distance by step")
 
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("d")

    axes[1].fill_between(
        time,
        mean - std,
        mean + std,
        alpha=0.1,
        color="r",
    )

    axes[1].plot(
        time, mean, "o-", color="r", label=f"w={args.horizon}, h={args.traj_len}"
    )
    fig.savefig(plot_file_name)   

    env.close()

if __name__ == "__main__":
    main()