from dataclasses import dataclass

import gym
import numpy as np

from core.ppo_agent import PPOAgent

@dataclass
class Trainer:
    environment_name: str
    batch_size: int
    num_of_epochs: int
    num_of_games: int
    learning_rate: float
    memory_size: int
    ma_period: int

    def __post_init__(self):
        self.env = gym.make(self.environment_name)
        self._create_agent()

    def _create_agent(self):
        self.agent = PPOAgent(num_of_actions = self.env.action_space.n, 
                    batch_size = self.batch_size, 
                    learning_rate = self.learning_rate, 
                    num_of_epochs = self.num_of_epochs, 
                    input_dims = self.env.observation_space.shape
                )
    
    def _print_step_info(self, game_no, score, avg_score, num_of_steps, learning_steps):
        print(f'Episode {game_no} Score {score} Avg. Score {avg_score} Time Steps {num_of_steps} Learning Steps {learning_steps}')

    def start_training(self):
        best_score = self.env.reward_range[0]
        score_history = []

        learning_steps = 0
        avg_score = 0
        num_of_steps = 0
        
        for game_no in range(self.num_of_games):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action, prob, value = self.agent.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)

                num_of_steps += 1
                score += reward
                
                self.agent.remember(state, action, prob, value, reward, done)

                if num_of_steps % self.memory_size == 0:
                    self.agent.learn()
                    learning_steps += 1
                
                state = new_state
            
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()
            
            self._print_step_info(game_no, score, avg_score, num_of_steps, learning_steps)
            
        self.score_history = score_history    