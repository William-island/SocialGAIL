import os
from time import time, sleep
from datetime import timedelta
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

class Tester:

    def __init__(self, env_test, algo, model_dir, seed=0):
        super().__init__()


        # Env for evaluation.
        self.env_test = env_test
        self.env_test.seed(2**31-seed)

        self.algo = algo
        self.model_dir = model_dir

    
    def test(self):
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        mean_return = 0.0
        frechet_distance = 0.0
        fde = 0.0

        for id in tqdm(test_ids):   
            state = self.env_test.reset(person_id=id)
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return 
            frechet_distance += self.env_test.compute_Frechet_Distance() 
            fde += self.env_test.compute_FDE()     

        mean_return /= len(test_ids)
        frechet_distance /= len(test_ids)
        fde /= len(test_ids)

        return mean_return, frechet_distance, fde
    


    def flow_test(self, flow_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        log_kde = 0

        for test_id in tqdm(flow_ids):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            log_kde += self.env_test.compute_log_kde(flow_ids)
        
        return log_kde
    

    def flow_test_ar(self, flow_ids):
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        log_kde = 0

        for test_id in tqdm(flow_ids):
            # init 
            self.env_test.reset(person_id=test_id)
            # render and compute
            log_kde += self.env_test.step_all(self.algo, flow_ids)
        
        return log_kde
    

    def states_test(self):
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        hdd = 0

        for test_id in tqdm(test_ids):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            hdd += self.env_test.compute_traj_hausdorff_distance()
        
        return hdd/len(test_ids)
    

    def short_test(self)->list:
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        FD_list = []

        for test_id in tqdm(test_ids[:300]):
            if test_id!=12323:
                state = self.env_test.reset(person_id=test_id)
                count = 0

                while (count<5):
                    action = self.algo.exploit(state)
                    state, _, done, _ = self.env_test.step(action)
                    count += 1

                FD_list.append(self.env_test.compute_short_FD())
        
        return FD_list
    
    def short_special(self, index)->list:
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')


        test_id = test_ids[index]
        state = self.env_test.reset(person_id=test_id)
        count = 0

        while (count<5):
            action = self.algo.exploit(state)
            state, _, done, _ = self.env_test.step(action)
            count += 1
        
        return self.env_test.get_short_traj()

    

    def long_test(self)->list:
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        FD_list = []

        for test_id in tqdm(test_ids[:]):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            FD_list.append(self.env_test.compute_Frechet_Distance())
        
        return FD_list
    

    def long_special(self, index)->list:
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')


        test_id = test_ids[index]
        state = self.env_test.reset(person_id=test_id)
        done = False

        while (not done):
            action = self.algo.exploit(state)
            state, _, done, _ = self.env_test.step(action)
        
        return self.env_test.get_short_traj()
    
    
    def collision_test(self)->list:
        # test id list
        with open('./gail_airl_ppo/test_people.pkl','rb') as f:
            test_ids = pickle.load(f)
        # load model dict
        model_state_dict = torch.load(self.model_dir)
        self.algo.actor.load_state_dict(model_state_dict['actor'])
        print(f'Load model {self.model_dir} success!')

        col_list = []

        for test_id in tqdm(test_ids[:]):
            state = self.env_test.reset(person_id=test_id)
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, _, done, _ = self.env_test.step(action)

            col_list.append(self.env_test.get_collisions())
        
        return col_list

