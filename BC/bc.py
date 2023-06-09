import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,input_dimension):
        super().__init__()
        self.layer1=nn.Linear(input_dimension,256)
        self.layer2=nn.Linear(256,256)
        self.layer3=nn.Linear(256,128)
        self.layer4=nn.Linear(128,128)
        self.out=nn.Linear(128,2)
        self.dropout=nn.Dropout(0.25)
    
    def forward(self,X):
        l1=F.relu(self.layer1(X))
        l2=F.relu(self.layer2(l1))
        l3=F.relu(self.layer3(l2))
        l4=F.relu(self.layer4(l3))
        out=self.out(l4)
        return out

class BC:
    def __init__(self, state_dim):
        self.actor = Actor(state_dim).to(device)
        self.actor.load_state_dict(torch.load('/home/william/SocialGAIL/BC/vanilla_goal.pt'))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()