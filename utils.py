import numpy as np
import pickle
from GNN_models.graph_utils import GraphData

class ExpertTraj:
    def __init__(self, args):

        if args.observation_type == 'graph':
            expert_dir = 'graph/'
        else:
            if args.with_last_speed:
                expert_dir = 'goal_lastSpeed/'
            else:
                expert_dir = 'goal/'
        print('Loading expert trajectories...')
        with open('./expert_traj/'+expert_dir+'states.pkl','rb') as f:
            self.exp_states = pickle.load(f)
        with open('./expert_traj/'+expert_dir+'actions.pkl','rb') as f:
            self.exp_actions = pickle.load(f)
        self.n_transitions = len(self.exp_actions)
        print("now expert lens: {}".format(self.n_transitions))
    
    def sample(self, batch_size):
        indexes = np.random.randint(0, self.n_transitions, size=batch_size)
        state, action = [], []
        for i in indexes:
            s = self.exp_states[i]
            a = self.exp_actions[i]
            state.append(s)
            action.append(a)
        return state, action
        #     state.append(np.array(s, copy=False))
        #     action.append(np.array(a, copy=False))
        # return np.array(state), np.array(action)


