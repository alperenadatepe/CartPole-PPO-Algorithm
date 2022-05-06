import numpy as np

from dataclasses import dataclass, field

@dataclass
class MemoryUnit:
    prob: float
    value: float
    action: int
    reward: float
    done: bool
    state: list[float] = field(default_factory=list)

@dataclass
class Memory:
    batch_size: int

    def __post_init__(self):
        self.memory_units = []

    def generate_batches(self):
        num_of_observations = len(self.memory_units)
        batch_start = np.arange(0, num_of_observations, self.batch_size)
        indices = np.arange(num_of_observations, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:(i + self.batch_size)] for i in batch_start]
    
        return batches
    
    def get_memory(self):
        self.states = np.array(list(map(lambda x: x.state, self.memory_units)))
        self.actions = np.array(list(map(lambda x: x.action, self.memory_units)))
        self.rewards = np.array(list(map(lambda x: x.reward, self.memory_units)))
        self.probs = np.array(list(map(lambda x: x.prob, self.memory_units)))
        self.values = np.array(list(map(lambda x: x.value, self.memory_units)))
        self.dones = np.array(list(map(lambda x: x.done, self.memory_units)))

        return self.states, self.actions, self.rewards, self.probs, self.values, self.dones

    def store_memory(self, state, action, prob, value, reward, done):
        memory_unit = MemoryUnit(prob=prob, 
                                value=value, 
                                action=action, 
                                reward=reward, 
                                done=done, 
                                state=state
                            )

        self.memory_units.append(memory_unit)

    def clear_memory(self):
        self.memory_units = []
