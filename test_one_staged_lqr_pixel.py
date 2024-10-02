"""
In this file, we hope to merge the implementation of CURL and EmbedLQR to achieve a new working scenario: EmbedLQR over Pixel-Based Control
# Author: XL
# Date: 2023.2.9
# Location: SFU MarsLab
"""
import os
import argparse
import yaml

import time
import torch
import dmc2gym
import json
import gym

import pybullet as p
from utils.buffer import *
from gym.wrappers import NormalizeObservation
from utils.utils import *
from algorithms.ae_sac import *
from algorithms.curl_sac import *
from algorithms.utils import *
from torch.utils.tensorboard import SummaryWriter


CONFIG_PATH = './config'


class FixLenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.step_count = 0
        self.env._max_episode_steps = max_steps
        self.done = False
    
    def reset(self, **kwargs):
        result =  super().reset(**kwargs)
        self.step_count = 0
        self.done = False
        return result
    
    def step(self, action):

        state, reward, done, info =  super().step(action)

        tmp = self.env.env.env.current_state
        cart_pos, cart_vel, pole_angle, pole_angular_vel = tmp[0], tmp[1], tmp[2], tmp[3]
        
        # Define the quadratic reward with respect to state and action
        # You can customize the coefficients for each term here
        quadratic_reward = -(
            1.0 * (cart_pos**2) +  # quadratic penalty on cart position
            #0.5 * (cart_vel**2) +  # quadratic penalty on cart velocity
            2.0 * (pole_angle**2) #+  # quadratic penalty on pole angle
            #0.5 * (pole_angular_vel**2) +  # quadratic penalty on pole angular velocity
            #0.1 * (action**2)  # quadratic penalty on action
        )
        
        if self.step_count >= self.env._max_episode_steps - 1:
            self.done = True
        else:
            self.step_count += 1
            self.done = False

        return state, quadratic_reward, self.done, info


def parse_args():
    parser = argparse.ArgumentParser(description='args', add_help=False)
    parser.add_argument('--config', type=str,
                        default='cheetah-run-embedlqr-state', help='Name of the config file')
    # parser.add_argument('--config', type=str,
    #                     default='cartpole-swingup', help='Name of the config file')
    args, unknown = parser.parse_known_args()
    return args


def setup_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def main():
    args = parse_args()
    #load yaml configuration
    with open(os.path.join(CONFIG_PATH, args.config+'.yaml')) as file:
        config = yaml.safe_load(file)

    set_seed_everywhere(config['seed'])
    ENV_NAME = 'cartpole'
    TASK_NAME = 'swingup'
    run_id = 'baseline'
    save_dir = f'saved/{TASK_NAME}_{ENV_NAME}/{run_id}'
    logdir = F'log/task_{TASK_NAME}_env_{ENV_NAME}/{run_id}'

    setup_directory(logdir)
    setup_directory(save_dir)
    
    writer = SummaryWriter(log_dir=logdir)
    MAX_STEPS = 200
    # set environment
    if config.get('domain_name') is not None and config.get('task_name') is not None:
        env = dmc2gym.make(
            domain_name=config['domain_name'],
            task_name=config['task_name'],
            seed=config['seed'],
            visualize_reward=False,
            from_pixels=(config['env']['encoder_type'] == 'pixel'),
            height=config['env']['pre_transform_image_size'],
            width=config['env']['pre_transform_image_size'],
            frame_skip=config['env']['action_repeat'])
    elif config.get('env_name') is not None:
        env = NormalizeObservation(gym.make(config['env_name']))
        env._max_episode_steps = MAX_STEPS
    env.seed(config['seed'])
    env = FixLenWrapper(env, MAX_STEPS)

    # stack several consecutive frames together
    if config['env']['encoder_type'] == 'pixel':
        env = FrameStack(env, k=config['env']['frame_stack'])
    

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = config['domain_name'] + '-' + config['task_name'] if config.get('domain_name') is not None and config.get('task_name') is not None else config['env_name']
    exp_name = env_name + '-' + ts + '-im' + str(config['env']['image_size']) +'-b'  \
    + str(config['train']['batch_size']) + '-s' + str(config['seed'])  + '-' + config['env']['encoder_type'] + '-' + str(time.time()).split(".")[0]
    config['work_dir'] = config['work_dir'] + '/'  + exp_name

    # make_dir(config['work_dir'])
    # video_dir = make_dir(os.path.join(config['work_dir'], 'video'))
    # model_dir = make_dir(os.path.join(config['work_dir'], 'model'))
    # buffer_dir = make_dir(os.path.join(config['work_dir'], 'buffer'))

    # video = VideoRecorder(video_dir if config['save_video'] else None)
    # print("video is initialized ...")

    # with open(os.path.join(config['work_dir'], 'args.json'), 'w') as f:
    #     json.dump(config, f, sort_keys=True, indent=4)

    # device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # replay buffer
    if config['env']['encoder_type'] == 'pixel':
        obs_shape = (3*config['env']['frame_stack'], config['env']['image_size'], config['env']['image_size'])
        pre_aug_obs_shape = (3*config['env']['frame_stack'],config['env']['pre_transform_image_size'],config['env']['pre_transform_image_size'])
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    action_shape = env.action_space.shape


    replay_buffer = ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=config['env']['replay_buffer_capacity'],
        batch_size=config['train']['batch_size'],
        device=device,
        image_size=config['env']['image_size'],
        from_pixel=(config['env']['encoder_type'] == 'pixel')
    )


    # make agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        config=config,
        device=device)
    
    
    # Logger
    L = Logger(config['work_dir'], use_tb=config['save_tb'])

    # training process
    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(config['train']['num_train_steps']):
        # evaluate agent periodically
        if step % config['eval']['eval_freq'] == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, None, config['eval']['num_eval_episodes'], L, step, config, writer=writer)
            # if config['save_model']:
            #     agent.save_curl(model_dir, step)
            #     agent.save(model_dir, step)
            # if config['save_buffer']:
            #     replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % config['log_interval'] == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % config['log_interval'] == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % config['log_interval'] == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < config['train']['init_steps']:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= config['train']['init_steps']:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == MAX_STEPS else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

