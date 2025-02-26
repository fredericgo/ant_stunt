import argparse
import datetime
import envs

import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

from rl.sac import SAC
from rl.replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_nrg",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=4000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--checkpoint_interval', type=int, default=500, 
                    help='checkpoint training model every # steps')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='checkpoint training model every # steps')
parser.add_argument('--eval_interval', type=int, default=100, 
                    help='checkpoint training model every # steps')
parser.add_argument('--rnd', type=bool, default=False, 
                    help='checkpoint training model every # steps')
parser.add_argument('--max_env_steps', type=int, default=50, 
                    help='checkpoint training model every # steps')
        
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = envs.create_env(args.env_name)

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_SAC_{args.env_name}_{args.policy}'
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)


def sample_trajectory(env, greedy=False, noise=0, render=False):
    states = []
    actions = []
    next_states = []
    rewards = []

    done = False
    state = env.reset()
    for t in range(args.max_env_steps):
        if render:
            env.render()

        states.append(state)
        if done:
            state = env.reset()

        action = agent.select_action(state[None])[0]
        
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards.append(reward)
        next_states.append(next_state)
    return np.stack(states), np.array(actions), np.stack(next_states), np.stack(rewards)


def evaluate_policy(env, eval_episodes=10, greedy=True, prefix='Eval', total_timesteps=0):
    
    all_states = []
    all_actions = []

    for index in range(eval_episodes):
        states, actions, _, rewards = sample_trajectory(env, noise=0, greedy=greedy, render=False)
        all_actions.extend(actions)
        all_states.append(states)
      
    all_states = np.stack(all_states)
    print('%s num episodes'%prefix, eval_episodes)
    print('%s reward'%prefix, np.mean(rewards))

    writer.add_scalar('%s/reward'%prefix,  np.mean(rewards), total_timesteps)
    
    return all_states

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == args.max_env_steps else float(not done)
        if not done:
            reward = 0.
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break
        
    if i_episode % args.log_interval == 0:
        writer.add_scalar('reward/train', episode_reward, i_episode)

    if i_episode % args.eval_interval == 0 and args.eval is True:
        evaluate_policy(env, total_timesteps=updates)
    
    if i_episode % args.checkpoint_interval == 0:
        agent.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {i_episode} episodes.")
        print("----------------------------------------")

env.close()

