import numpy as np
from ttictoc import TicToc
from typing import Union, Tuple

from memory.sumtree import SumTree
from memory.common import ExperienceSamples
from memory.uniform_memory import UniformExperienceBuffer
from utils.utilities import StackSet
from utils.wrappers import LazyFrames


class PrioritizedExperienceBuffer(UniformExperienceBuffer):
    def __init__(self, capacity, n_step, eps=0.01, alpha=0.6, beta0=0.4,
                 betasteps=1000, gamma=0.99, ir_prob=0.2, n_policies=1, normalize_by_max=True, **kwargs):
        super(PrioritizedExperienceBuffer, self).__init__(capacity, n_step, gamma, **kwargs)
        self.eps = eps
        self.alpha = alpha
        self.beta0 = beta0
        self.min_error = 0.0
        self.max_error = 1.0
        self.beta_add = (1.0 - self.beta0) / betasteps
        self.ir_prob = ir_prob
        self.normalize_by_max = normalize_by_max

        self.sumtree = [SumTree(capacity) for _ in range(n_policies)]
        self.lru_idxs = StackSet(range(capacity), maxlen=capacity)

    def append(
            self, state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]] = None,
            action: Union[np.ndarray, int] = None, reward: int = 0, done: bool = False,
            next_state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]] = None,
            skip_step: int = 0, p_idx: int = 0, prob=None, expert: bool = False,
            **kwargs) -> None:
        # transition_id = 4 * p_idx + skip_step
        transition_id = skip_step
        last_n_transitions = self.last_n_transitions[transition_id]
        last_n_transitions.append(dict(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            policy=p_idx,
            expert=expert,
            **kwargs
        ))
        if len(last_n_transitions) == self.n_step:
            self.add_transition(last_n_transitions, done, prob, p_idx=p_idx)

    def add_transition(self, transitions, done, priority=None, p_idx=0):
        if not transitions:
            return
        sumtree = self.sumtree[p_idx]
        if self.is_full and self.ir_prob > 0 and np.random.rand() < self.ir_prob:
            idx = sumtree.sample_low()
            idx = self.lru_idxs.pop(val=idx)
        else:
            idx = self.lru_idxs.pop()

        sumtree.append(idx=idx, priority=priority)
        self.entries = min(self.entries+1, self.capacity)

        self.data[idx] = list(transitions)

        # Remove first transition and append
        if done:
            del transitions[0]
            self.add_transition(transitions, done, priority, p_idx)

    def sample(self, n, p_idx=0) -> Tuple[ExperienceSamples, np.ndarray]:
        sumtree = self.sumtree[p_idx]
        self.beta0 = min(1., self.beta0 + self.beta_add)
        idxs, priorities = sumtree.prioritized_sample(n)
        samples = self.data[idxs].tolist()
        samples = ExperienceSamples(samples, n_step=self.n_step, gamma=self.gamma)
        is_weight = np.power(sumtree.entries * priorities, -self.beta0)
        is_weight /= is_weight.max()
        return samples, is_weight

    def _priority_from_errors(self, errors: np.ndarray):
        if self.normalize_by_max:
            # Not in paper, but might help to keep td errors manageable.
            errors = errors.clip(0.0) / errors.max()
        return np.power(errors + self.eps, self.alpha)

    def update_priorities(self, errors, p_idx=0):
        sumtree = self.sumtree[p_idx]
        priorities = self._priority_from_errors(errors)
        sumtree.update_weights(priorities)


if __name__ == "__main__":
    num_elements = 16
    sample_size = 2

    elements = np.arange(num_elements)
    # probabilities should total_sum to 1
    probabilities = np.linspace(start=1, stop=num_elements, num=num_elements, endpoint=True)
    # probabilities /= np.sum(probabilities)

    memory = PrioritizedExperienceBuffer(num_elements, n_step=1, betasteps=100)

    for ele in probabilities:
        memory.append()
    print(memory.sumtree[0].tree)
    with TicToc():
        for _ in range(10000):
            data, weights = memory.sample(sample_size)
            print(weights)
            new_weights = np.random.uniform(low=0.5, size=weights.shape)
            memory.update_priorities(new_weights)
    print(memory.sumtree[0].tree[:memory.capacity-1])
    print(memory.sumtree[0].tree[-memory.capacity:])
