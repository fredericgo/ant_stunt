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
parser.add_argument('--env-name', default="ant_jump",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--initial_state_file', type=str, default=None, required=True,
                    help='initial state')
parser.add_argument('--z',  default=1.,
                    help='height')

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
parser.add_argument('--video_file_name', type=str, default=None, 
                    help='video file name')
args = parser.parse_args()


def load_policy(policy, model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def render_trajectory(policy, env, writer):

    state = env.reset()
    done = False
    
    for t in range(args.traj_len):
        title_font = ImageFont.truetype('dejavu/DejaVuSerif-Bold.ttf', 50)
        title_text = str(t)
        
        x = env.render(mode="rgb_array")

        img = Image.fromarray(x)
        draw = ImageDraw.Draw(img)

        draw.text((15,15), title_text, (237, 230, 211), font=title_font)
        img = np.array(img)

        writer.append_data(img)

        if done:
            state = env.reset()

        state = torch.FloatTensor(state).unsqueeze(0)
        _, _, action = policy.sample(state)
        action = action.detach().cpu().numpy()[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)
        state, r, done, info = env.step(action)
        #print(state.shape)
        print(t, r, state[0])
        #print(t, r, done, state[15:18], state[18:21])


def main():
    # Environment
    env = envs.create_env(
            args.env_name, 
            initial_state_file=args.initial_state_file,
            z=args.z,
            init_threshold=0.2,
            threshold=0.1,
            max_episode_steps=args.traj_len)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = GaussianPolicy(env.num_features, env.action_space.shape[0], args.hidden_size, env.action_space)
    load_policy(policy, args.model_dir)
    
    if args.video_file_name:
        writer = imageio.get_writer(args.video_file_name, fps=30) 
    else:
        writer = None

    for _ in range(args.num_epochs):
        render_trajectory(policy, env, writer)

    env.close()

if __name__ == "__main__":
    main()