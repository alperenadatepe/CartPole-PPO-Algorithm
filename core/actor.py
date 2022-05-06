import os
from dataclasses import dataclass

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from torchinfo import summary

@dataclass(eq=False)
class ActorNetwork(nn.Module):
    num_of_actions: int
    input_dims: tuple
    learning_rate: float
    fc1_dims: int = 256
    fc2_dims: int = 256

    def _create_actor(self):
        self.actor = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, self.num_of_actions),
            nn.Softmax(dim = -1)
        )

    def _create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _create_device(self):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def __post_init__(self):
        super(ActorNetwork, self).__init__()

        self._create_actor()
        self._create_optimizer()
        self._create_device()

    def forward(self, state):
        prob_dist = self.actor(state)
        prob_dist = Categorical(prob_dist)
        
        return prob_dist
    
    def get_summary(self):
        model_summary = summary(self.actor, input_size=self.input_dims, verbose=0)

        return str(model_summary)
    
    def save_checkpoint(self):
        self.checkpoint_file = os.path.join('checkpoints/actor_torch_ppo.pt')
        T.save(self.state_dict(), self.checkpoint_file)
