from abc import ABC, abstractmethod
import numpy as np
from collections import OrderedDict
from typing import List, Dict


def reward_discount(sample: List[Dict], gamma=0.99):
    if "discounted_reward" in sample[0]:
        return sample[0]["discounted_reward"]
    discounted_reward = sum([step["reward"] * gamma**idx for idx, step in enumerate(sample)])
    sample[0]["discounted_reward"] = discounted_reward
    return discounted_reward


class ReplayBufferAbstract(ABC):
    demo_samples: int

    @abstractmethod
    def set_pretrain_phase(self, pretraining: bool):
        pass

    @abstractmethod
    def append(self, state=None, action=None, reward=0, done=False, next_state=None,
               skip_step=0, p_idx=0, prob=None, expert=False, **kwargs):
        pass

    @abstractmethod
    def sample(self, n, demo_fraction=0.5, p_idx=0):
        pass

    @abstractmethod
    def update_priorities(self, errors, p_idx=0):
        pass

    @property
    def is_full(self):
        return False

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def stop_current_episode(self, env_id=0, p_idx=0):
        pass


class ExperienceSamples:
    """
    Object to store a batch of experiences.
    Centralizes nearly all batch processing to small functions.
    """
    def __init__(self, samples: List[List[Dict]], n_step=10, gamma=0.99, state_cache={}):
        self.samples = samples
        self.n_step = n_step
        self.gamma = gamma
        self.is_branched: bool = isinstance(samples[0][0]["action"], (OrderedDict, Dict))
        if not state_cache:
            state_cache['state'] = np.zeros_like(np.asarray(samples[0][0]["state"]))
        self._empty_state = state_cache['state']

    @property
    def states(self):
        for sample in self.samples:
            yield np.asarray(sample[0]["state"])

    @property
    def next_states(self):
        for sample in self.samples:
            yield np.asarray(sample[0]["next_state"])

    @property
    def nth_states(self):
        for sample in self.samples:
            yield np.asarray(sample[-1]["next_state"]) if len(sample) >= self.n_step else self._empty_state

    @property
    def actions(self):
        if not self.is_branched:
            return np.array([sample[0]["action"] for sample in self.samples])
        else:
            return np.array([list(sample[0]["action"].values()) for sample in self.samples])

    @property
    def non_terminals(self):
        return np.array([not sample[0]["done"] for sample in self.samples])

    @property
    def nth_non_terminals(self):
        return np.array([not sample[-1]["done"] for sample in self.samples])

    @property
    def rewards(self):
        return np.array([sample[0]["reward"] for sample in self.samples])

    @property
    def nth_rewards(self):
        return np.array([reward_discount(sample) for sample in self.samples])

    @property
    def experts(self):
        return np.array([sample[0]["expert"] for sample in self.samples])

    @property
    def expert_scales(self):
        return np.array([sample[0]["expert_scale"] for sample in self.samples])

    @property
    def policies(self):
        return np.array([sample[0]["policy"] for sample in self.samples])

