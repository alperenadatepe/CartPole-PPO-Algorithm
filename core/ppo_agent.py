from dataclasses import dataclass

import numpy as np
import torch as T

from .memory import Memory
from .actor import ActorNetwork
from .critic import CriticNetwork

@dataclass
class PPOAgent:
    num_of_actions: int
    input_dims: tuple
    gamma: float = 0.99
    learning_rate: float = 0.0003
    lambda_value: float = 0.95
    policy_clip: float = 0.2
    batch_size: float = 64
    num_of_epochs: int = 10

    def __post_init__(self):
        self.actor = ActorNetwork(self.num_of_actions, self.input_dims, self.learning_rate)
        self.critic = CriticNetwork(self.input_dims, self.learning_rate)
        self.memory = Memory(self.batch_size)

    def _calculate_advantage(self, rewards, dones, values):
        advantage = np.zeros(len(rewards), dtype=np.float32)

        for i in range(len(rewards) - 1):
            discount = 1
            advantage_t = 0
            for t in range(i, len(rewards) - 1):
                delta_t = (rewards[t] + self.gamma * values[t + 1] * (1 - int(dones[t])) - values[t])
                advantage_t += discount * delta_t
                discount *= self.gamma * self.lambda_value

            advantage[i] = advantage_t

        advantage_tensor = T.tensor(advantage).to(self.actor.device)

        return advantage_tensor

    def _calculate_actor_loss(self, actions_tensor, old_probs_tensor, prob_dist, advantage_tensor_batched):
        new_probs_tensor = prob_dist.log_prob(actions_tensor)
        prob_ratio = new_probs_tensor.exp() / old_probs_tensor.exp()

        loss_cpi = advantage_tensor_batched * prob_ratio
        clipped_loss_cpi = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage_tensor_batched

        actor_loss = -T.min(loss_cpi, clipped_loss_cpi).mean()

        return actor_loss
    
    def _calculate_critic_loss(self, critic_value, advantage_tensor_batched, values_tensor_batched):
        returns = advantage_tensor_batched + values_tensor_batched
        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()

        return critic_loss

    def _calculate_total_loss(self, actions_tensor, old_probs_tensor, prob_dist, critic_value, advantage_tensor_batched, values_tensor_batched):
        actor_loss = self._calculate_actor_loss(actions_tensor, old_probs_tensor, prob_dist, advantage_tensor_batched)
        critic_loss = self._calculate_critic_loss(critic_value, advantage_tensor_batched, values_tensor_batched)

        total_loss = actor_loss + 0.5 * critic_loss

        return total_loss
    
    def _reset_grads(self):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

    def _loss_backward(self, total_loss):
        total_loss.backward()
    
    def _optimizer_step(self):
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def _process_batches(self, batches, advantage_tensor, values, states, actions, old_probs):
        values_tensor = T.tensor(values).to(self.actor.device)

        for batch in batches:
            states_tensor = T.tensor(states[batch], dtype=T.float).to(self.actor.device)
            old_probs_tensor = T.tensor(old_probs[batch]).to(self.actor.device)
            actions_tensor = T.tensor(actions[batch]).to(self.actor.device)

            prob_dist = self.actor.forward(states_tensor)
            critic_value = T.squeeze(self.critic.forward(states_tensor))

            total_loss = self._calculate_total_loss(actions_tensor, old_probs_tensor, prob_dist, critic_value, advantage_tensor[batch], values_tensor[batch])
            
            self._reset_grads()
            self._loss_backward(total_loss)
            self._optimizer_step()
       
    def remember(self, state, action, prob, value, reward, done):
        self.memory.store_memory(state, action, prob, value, reward, done)

    def choose_action(self, state):
        state = T.tensor([state], dtype=T.float).to(self.actor.device)

        prob_dist = self.actor(state)
        value = self.critic(state)
        action = prob_dist.sample()

        probs = T.squeeze(prob_dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.num_of_epochs):
            batches = self.memory.generate_batches()
            states, actions, rewards, old_probs, values, dones = self.memory.get_memory()

            advantage = self._calculate_advantage(rewards, dones, values)

            self._process_batches(batches, advantage, values, states, actions, old_probs)
    
        self.memory.clear_memory()               
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()