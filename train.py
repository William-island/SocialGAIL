import torch
import gym
import numpy as np
from BC.bc import BC
import matplotlib.pyplot as plt
import argparse
from gym_env import CrowdEnv
from GAIL import GAIL
import time
from GNN_models.graph_utils import GraphData



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

class Logger():

    def __init__(self, path):
        self.path = path
        with open(self.path, 'w') as f:
            pass

    def write_log(self,log):
        with open(self.path, 'a') as f:
            f.write(log+'\n')





def train_socialgail(args):
    env = CrowdEnv(args)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    agent = GAIL(args)

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

    logger_all=Logger('log_all.txt')
    logger_res=Logger('log.txt')

    current_reward = 0

    while(steps < args.total_steps):
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

        if done or t > args.max_timesteps:
            t = 0
            n_eposides += 1
            # frechet_disance += env.compute_Frechet_Distance()
            # fde += env.compute_FDE()
            # dtw += env.compute_DTW()
            fre = env.compute_Frechet_Distance()
            fd = env.compute_FDE()
            dt = env.compute_DTW()

            frechet_disance += fre
            fde += fd
            dtw += dt
            delta_reward = total_reward-current_reward
            current_reward = total_reward
            log_content="\t eposide: {}\t Reward: {}\tFrechet Distance: {:.2f}\t FDE: {:.2f}\t DTW: {:.2f}\t      ID: {}".format \
                (n_eposides, delta_reward, fre, fd, dt, env.get_agent_id())
            logger_all.write_log(log_content)
            print(log_content)

            state = env.reset()

        if steps%args.training_interval==0:
            epoch += 1
            # update agent n_iter times
            if args.observation_type == 'graph':
                agent.update_online_graph(args.n_iter, args.batch_size)
            else:
                agent.update_online(args.n_iter, args.batch_size)
                
            avg_reward = int(total_reward/n_eposides)
            avg_frechet_distance = frechet_disance/n_eposides
            avg_fde = fde/n_eposides
            avg_dtw = dtw/n_eposides
            log_content="Epoch: {}\tAvg Reward: {}\tFrechet Distance: {:.2f}\t  Avg FDE: {:.2f}\t  Avg DTW: {:.2f}\t  Num eposides: {}".format \
                  (epoch, avg_reward, avg_frechet_distance, avg_fde, avg_dtw, n_eposides)
            logger_all.write_log(log_content)
            logger_res.write_log(log_content)
            print(log_content)
            
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

            current_reward = 0
        
        # if continuous action space; then decay action std of ouput action distribution
            if steps % args.action_std_decay_freq == 0:
                agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)

    end = time.time()
    mins, secs = compute_time(start, end)
    log_content=f'Total Time: {mins}m {secs}s'
    logger_all.write_log(log_content)
    logger_res.write_log(log_content)
    print(log_content)
    
    ## plot and save graph
    draw_result(epochs, rewards, 'reward')
    draw_result(epochs, frechet_disance_list, 'Frechet Distance')
    draw_result(epochs, fde_list, 'FDE')
    draw_result(epochs, dtw_list, 'DTW')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for SocialGAIL')
    parser.add_argument('--env_name',default="CrowdEnv")
    parser.add_argument('--dataset_path',default="./datasets/gc_interpolated_trajectory.pkl")
    parser.add_argument('--frame_interval',default=10)
    parser.add_argument('--time_interval',default=0.4)
    parser.add_argument('--regions',default=16)
    parser.add_argument('--radius',default=6.0)
    parser.add_argument('--map_size_bound',default=[-10,40,-20,50])         # [low_x, high_x, low_y, high_y] (int)
    parser.add_argument('--with_last_speed',default=True)
    parser.add_argument('--observation_type',default='graph')               # 'radar' / 'graph'
    parser.add_argument('--graph_obs_past_len',default=5)
    parser.add_argument('--padd_to_number',default=60)                      # the max number of people in radius to form a mini-batch
    parser.add_argument('--graph_feature_len',default=5)
    parser.add_argument('--output_len',default=2)
    # training hyperparamater
    parser.add_argument('--max_timesteps',default=800)                      # max time steps in one episode
    parser.add_argument('--total_steps',default=1000000)                    # int(1.6e6) 800000
    parser.add_argument('--training_interval',default=2048)                    # int(1.6e6) 800000
    parser.add_argument('--n_iter',default=10)                              # updates per epoch
    parser.add_argument('--batch_size',default=64)                          # num of transitions sampled from expert
    parser.add_argument('--action_std_decay_rate',default=0.05)             # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    parser.add_argument('--min_action_std',default=0.1)                     # minimum action_std (stop decay after action_std <= min_action_std)
    parser.add_argument('--action_std_decay_freq',default=int(2.5e5))       # action_std decay frequency (in num timesteps)
    args = parser.parse_args()



    train_socialgail(args)