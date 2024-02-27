import os
import argparse
from datetime import datetime
import torch
import pickle
import copy

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import *
from gail_airl_ppo.algo import ALGOS,PPO
from gail_airl_ppo.tester import Tester

import concurrent 


def run_once(args, model_subdir,cuda_device):
    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device(cuda_device if args.cuda else "cpu"),  # "cuda:0"
        seed=args.seed
    )

    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.model_dir+'/'+model_subdir
    )

    mean_return, frechet_distance, fde = tester.test()

    return model_subdir, mean_return, frechet_distance, fde

# test FDE FD
def test_fde_fd(args):
    # test models
    models_list = sorted(os.listdir(args.model_dir), key = lambda x:int(x[4:-3]))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_models) as executor:
        to_dos = []
        res_list = []
        # submit
        for i in range(args.num_models):
            model_subdir = models_list[-args.num_models+i]
            cuda_device = "cuda:" + str(i%4)
            future = executor.submit(run_once, args, model_subdir, cuda_device)
            to_dos.append(future)
        # wait all to finish
        for future in concurrent.futures.as_completed(to_dos):
            res_list.append(future.result())
        # show res
        res_list = sorted(res_list, key=lambda tup:int(tup[0][4:-3]))
        for tup in res_list:
            model_dir, mean_return, frechet_distance, fde = tup
            print(f'Test Model: {model_dir}  '
              f'    Return: {mean_return:<5.1f}   '
              f'Frechet Distance: {frechet_distance:<5.2f}   '
              f'FDE: {fde:<5.2f}   ')

# test FDE FD for one pt
def test_fde_fd_one(args):
    # test model
    model_dir, mean_return, frechet_distance, fde = run_once(args, args.flow_model_path.split('/')[-1], 'cuda:1')
    print(f'Test Model: {model_dir}  '
        f'    Return: {mean_return:<5.1f}   '
        f'Frechet Distance: {frechet_distance:<5.2f}   '
        f'FDE: {fde:<5.2f}   ')
            





# test global similarity
def test_global_similarity(args):
    # test flow
    with open(args.flow_path,'rb') as f:
        flow_ids = pickle.load(f)

    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device(args.device if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    log_kde = tester.flow_test(flow_ids)
    
    print(f"{args.observation_type}   {args.flow_path[-9:-4]}  log KDE: {log_kde}")


# test global similarity with all rendered by actor
def test_global_similarity_ar(args):
    # test flow
    with open(args.flow_path,'rb') as f:
        flow_ids = pickle.load(f)

    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device(args.device if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    log_kde = tester.flow_test_ar(flow_ids)

    print(f"{args.observation_type}   {args.flow_path[-9:-4]}  log KDE: {log_kde}")


def test_global_similarity_parallel(args):
    # test flows
    flow_list = os.listdir(args.flow_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.flow_num) as executor:
        # submit
        for i in range(args.flow_num):
            flow_path = args.flow_dir+flow_list[i]
            cuda_device = "cuda:" + str(2+i%2)

            new_args = copy.copy(args)
            new_args.device = cuda_device
            new_args.flow_path = flow_path

            executor.submit(test_global_similarity_ar, new_args)  # test_global_similarity_ar


# # test global similarity
# def test_global_similarity(args):
#     # test flow
#     with open(args.flow_path,'rb') as f:
#         flow_ids = pickle.load(f)

#     env_test = make_env(args,'Test')
    
#     algo = PPO(
#         graph_feature_channels=args.graph_obs_past_len,
#         state_shape=env_test.observation_space.shape,
#         action_shape=env_test.action_space.shape,
#         device=torch.device(args.device if args.cuda else "cpu"),  # "cuda:1"
#         seed=args.seed
#     )


#     tester = Tester(
#         env_test=env_test,
#         algo=algo,
#         model_dir = args.flow_model_path
#     )



# test short similarity
def test_short_similarity(args):
    print('Computing short Similarity!\n')

    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device('cuda:1' if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    FD_list = tester.short_test()

    print(FD_list)

    with open('./saved_case/short_paths/socialGAIL.pkl','wb') as f:
        pickle.dump(FD_list,f)
    

# test short similarity
def test_short_special(args):
    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device('cuda:1' if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )


    traj_new = tester.short_special(args.special_index)

    with open('./saved_case/short_paths/trajectories/socialGAIL.pkl','wb') as f:
        pickle.dump(traj_new,f)







# test short similarity
def test_long_similarity(args):
    print('Computing long Similarity!\n')

    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device('cuda:1' if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    FD_list = tester.long_test()

    print(FD_list)

    with open('./saved_traj/socialGAIL.pkl','wb') as f:
        pickle.dump(FD_list,f)


# test long similarity
def test_long_special(args):
    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device('cuda:1' if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )


    traj_new = tester.long_special(args.special_index)

    with open('./saved_traj/render/trajectories/socialGAIL.pkl','wb') as f:
        pickle.dump(traj_new,f)




def test_collision(args):
    print('Computing collision!\n')

    env_test = make_env(args,'Test')
    
    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device('cuda:1' if args.cuda else "cpu"),  # "cuda:1"
        seed=args.seed
    )


    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    col_list = tester.collision_test()

    print(col_list)

    with open('./saved_traj/socialGAIL_collision.pkl','wb') as f:
        pickle.dump(col_list,f)



# test states similarity
def test_states_similarity(args):
    print('Computing State Similarity!\n')

    env_test = make_env(args,'Test')

    algo = PPO(
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device(args.device if args.cuda else "cpu"),  # "cuda:0"
        seed=args.seed
    )
    

    tester = Tester(
        env_test=env_test,
        algo=algo,
        model_dir = args.flow_model_path
    )

    hdd = tester.states_test()
    
    print(f"Hausdorff: {hdd}")



if __name__ == '__main__':
    # torch multiprocess
    torch.multiprocessing.set_start_method('spawn')

    p = argparse.ArgumentParser()

    # test fde fd
    p.add_argument('--model_dir', type=str, default='./logs/good_models/socialgail/best.pt')     #'./logs/CrowdEnv/gail/seed0-20230819-1450/model')
    p.add_argument('--num_models', type=int, default=20)

    # test flow
    p.add_argument('--flow_model_path', type=str, default='./logs/good_models/socialgail/best.pt')         #'./logs/good_models/socialgail/best.pt')
    p.add_argument('--flow_path', type=str, default='./global_flows/flow3.pkl')
    p.add_argument('--flow_dir', type=str, default='./global_flows/')
    p.add_argument('--flow_num', type=int, default=5)

    # special id
    p.add_argument('--special_index', type=int, default=15)


    # training parser
    p.add_argument('--rollout_length', type=int, default=2048)
    p.add_argument('--num_steps', type=int, default=int(7e5))   # 5.8e5 | 1.2e6
    p.add_argument('--eval_interval', type=int, default=5000)
    p.add_argument('--env_id', type=str, default="CrowdEnv")
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', default=True, action='store_true')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--seed', type=int, default=0)

    # Coustormed parser
    p.add_argument('--dataset_path',default="./gail_airl_ppo/crowd_env/datasets/GC_all.pkl")  # GC_all.pkl
    p.add_argument('--frame_interval',default=10)
    p.add_argument('--time_interval',default=0.4)
    p.add_argument('--regions',default=16)
    p.add_argument('--radius',default=6.0)
    p.add_argument('--map_size_bound',default=[-10,40,-20,50])         # [low_x, high_x, low_y, high_y] (int)
    p.add_argument('--with_last_speed',default=False)
    p.add_argument('--entity_type',default=False)
    p.add_argument('--observation_type',default='graph')               # 'radar' / 'graph'
    p.add_argument('--graph_obs_past_len',default=5)
    p.add_argument('--padd_to_number',default=60)                      # the max number of people in radius to form a mini-batch
    p.add_argument('--graph_feature_len',default=5)
    p.add_argument('--output_len',default=2)
    # other hyperparamater
    
    args = p.parse_args()

    
    test_states_similarity(args)
    # test_global_similarity_ar(args)
    # test_global_similarity(args)
    # test_global_similarity_parallel(args)
    # test_short_similarity(args)
    # test_short_special(args)
    # test_long_similarity(args)
    # test_long_special(args)
    # test_collision(args)
    # test_fde_fd_one(args)