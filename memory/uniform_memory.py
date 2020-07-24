import numpy as np
from collections import deque, defaultdict
from typing import Union, Tuple

from memory.common import ReplayBufferAbstract, ExperienceSamples
from utils.wrappers import LazyFrames


class UniformExperienceBuffer(ReplayBufferAbstract):
    def __init__(self, capacity, n_step, gamma=0.99, *args, **kwargs):
        self.entries = 0
        self.idx = 0

        self.gamma = gamma
        self.capacity = capacity
        self.n_step = n_step

        self.data = np.empty(self.capacity, dtype=object)

        self.last_n_transitions = defaultdict(
            lambda: deque([], maxlen=n_step))

    def set_pretrain_phase(self, pretraining: bool):
        """
        Only needed for demo memory
        :param pretraining:
        :return: None
        """
        pass

    def append(self,
               state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]]= None,
               action: Union[np.ndarray, int] = None,
               reward: int = 0,
               done: bool = False,
               next_state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]] = None,
               skip_step: int = 0, p_idx: int = 0, expert: bool = False,
               **kwargs) -> None:
        last_n_transitions = self.last_n_transitions[skip_step]
        last_n_transitions.append(dict(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            policy=p_idx,
            expert=expert
        ))
        if len(last_n_transitions) == self.n_step:
            self.add_transition(last_n_transitions, done)

    def add_transition(self, transitions, done):
        self.entries = min(self.entries+1, self.capacity)
        self.data[self.idx] = list(transitions)
        self.idx = (self.idx + 1) % self.capacity

        # Remove first transition and append
        if done:
            del transitions[0]
            if transitions:
                self.add_transition(transitions, done)

    def sample(self, n, **kwargs) -> Tuple[ExperienceSamples, None]:
        idxs = np.random.randint(low=0, high=self.entries, size=n)
        samples = self.data[idxs].tolist()
        samples = ExperienceSamples(samples, n_step=self.n_step, gamma=self.gamma)
        return samples, None

    def update_priorities(self, errors, **kwargs):
        """
        Should not do anything in uniform sampling, makes code simpler by not requiring another if.
        :param errors:
        :return: None
        """
        pass

    @property
    def is_full(self):
        return self.data[-1] is not None

    def __len__(self):
        return self.entries

    def stop_current_episode(self, env_id=0, p_idx=0):
        last_n_transitions = self.last_n_transitions[env_id]
        self.add_transition(last_n_transitions, done=True)
