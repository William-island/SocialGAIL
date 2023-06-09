import torch
import gym
import numpy as np
from BC.bc import BC
import matplotlib.pyplot as plt
import argparse
from gym_env import CrowdEnv
from GAIL import GAIL
import time

def draw_result(x, y, metric_name):
    plt.cla()
    plt.plot(x, y)
    plt.xlabel('Epochs')
    plt.ylabel('{}'.format(metric_name))
    plt.title('{}'.format(metric_name))
    plt.savefig('./gail_figure/new/{}.png'.format(metric_name))

def compute_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_socialgail(args):
    ######### Hyperparameters #########
    env_name = "CrowdEnv"
    max_timesteps = 800        # max time steps in one episode
    total_steps = 1000000 # int(1.6e6) 800000
    training_interval = 2048
    n_iter = 10                # updates per epoch
    batch_size = 64            # num of transitions sampled from expert
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    ###################################

    env = CrowdEnv(args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = GAIL(args, state_dim, action_dim)

    # graph logging variables:
    epochs = []
    rewards = []
    frechet_disance_list = []
    fde_list = []
    dtw_list = []
    
    # training procedure
    epoch = 0
    steps = 0
    state = env.reset()
    t = 0
    total_reward = 0
    frechet_disance = 0
    fde = 0
    dtw = 0
    n_eposides = 0

    start = time.time()

    while(steps < total_steps):
        steps += 1
        t += 1
        # evaluate in environment

        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # saving reward and is_terminals
        # agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)


        state=next_state
        total_reward += reward

        if done or t > max_timesteps:
            t = 0
            n_eposides += 1
            frechet_disance += env.compute_Frechet_Distance()
            fde += env.compute_FDE()
            dtw += env.compute_DTW()

            state = env.reset()

        if steps%training_interval==0:
            epoch += 1
            # update agent n_iter times
            agent.update_online(n_iter, batch_size)
                
            avg_reward = int(total_reward/n_eposides)
            avg_frechet_distance = frechet_disance/n_eposides
            avg_fde = fde/n_eposides
            avg_dtw = dtw/n_eposides
            print("Epoch: {}\tAvg Reward: {}\tFrechet Distance: {:.2f}\t  Avg FDE: {:.2f}\t  Avg DTW: {:.2f}\t  Num eposides: {}".format \
                  (epoch, avg_reward, avg_frechet_distance, avg_fde, avg_dtw, n_eposides))
            
            # add data for graph
            epochs.append(epoch)
            rewards.append(avg_reward)
            frechet_disance_list.append(avg_frechet_distance)
            fde_list.append(avg_fde)
            dtw_list.append(avg_dtw)

            total_reward = 0
            frechet_disance = 0
            fde = 0
            dtw = 0
            n_eposides = 0
        
        # if continuous action space; then decay action std of ouput action distribution
            if steps % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)

    end = time.time()
    mins, secs = compute_time(start, end)
    print(f'Total Time: {mins}m {secs}s')
    
    ## plot and save graph
    draw_result(epochs, rewards, 'reward')
    draw_result(epochs, frechet_disance_list, 'Frechet Distance')
    draw_result(epochs, fde_list, 'FDE')
    draw_result(epochs, dtw_list, 'DTW')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for SocialGAIL')
    parser.add_argument('--dataset_path',default="./datasets/gc_interpolated_trajectory.pkl")
    parser.add_argument('--frame_interval',default=10)
    parser.add_argument('--time_interval',default=0.4)
    parser.add_argument('--regions',default=16)
    parser.add_argument('--radius',default=6.0)
    parser.add_argument('--map_size_bound',default=[-10,40,-20,50]) # [low_x, high_x, low_y, high_y] (int)
    parser.add_argument('--with_last_speed',default=True)
    args = parser.parse_args()

    train_socialgail(args)