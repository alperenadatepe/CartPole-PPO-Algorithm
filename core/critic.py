import os
from dataclasses import dataclass

import torch as T
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

@dataclass(eq=False)
class CriticNetwork(nn.Module):
    input_dims: tuple
    learning_rate: float
    fc1_dims: int = 256
    fc2_dims: int = 256

    def _create_critic(self):
        self.critic = nn.Sequential(
            nn.Linear(*self.input_dims, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

    def _create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _create_device(self):
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def __post_init__(self):
        super(CriticNetwork, self).__init__()

        self._create_critic()
        self._create_optimizer()
        self._create_device()

    def forward(self, state):
        value = self.critic(state)

        return value

    def get_summary(self):
        model_summary = summary(self.critic, input_size=self.input_dims, verbose=0)

        return str(model_summary)  
    
    def save_checkpoint(self):
        self.checkpoint_file = os.path.join('checkpoints/critic_torch_ppo.pt')
        T.save(self.state_dict(), self.checkpoint_file)