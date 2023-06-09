import torch
import gym
import numpy as np
from BC.bc import BC
import matplotlib.pyplot as plt
from gym_env import CrowdEnv
import argparse
from tqdm import *
from statistics import mean

def test_bc(args):
    env = CrowdEnv(args)
    policy = BC(3*args.regions+2)
    reward_list = []
    fde_list = []
    fd_list = []
    dtw_list = []
    for episode in trange(5):
        sum_reward = 0
        state = env.reset()
        for t in range(500):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            sum_reward += reward
            if done:
                env.generate_gif()
                fde = env.compute_FDE()
                fd = env.compute_Frechet_Distance()
                dtw = env.compute_DTW()
                break
        fde_list.append(fde)
        fd_list.append(fd)
        dtw_list.append(dtw)
        reward_list.append(sum_reward)
    print(f'Frechet Distance: {mean(fd_list)}')  
    print(f'Average FDE: {mean(fde_list)}')
    print(f'Average DTW: {mean(dtw_list)}')
    print(f'Average reward: {sum(reward_list)/len(reward_list)}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for SocialGAIL')
    parser.add_argument('--dataset_path',default="./datasets/gc_homo_trajectory.pkl")
    parser.add_argument('--regions',default=16)
    parser.add_argument('--radius',default=6)
    parser.add_argument('--frame_interval',default=20)
    parser.add_argument('--time_interval',default=0.8)
    parser.add_argument('--map_size_bound',default=[-10,40,-20,50]) # [low_x, high_x, low_y, high_y] (int)
    parser.add_argument('--with_last_speed',default=False)
    args = parser.parse_args()



    test_bc(args)