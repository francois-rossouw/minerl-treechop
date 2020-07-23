import numpy as np
from typing import Union, Tuple

from memory.common import ReplayBufferAbstract, ExperienceSamples
from memory.prioritized_memory import PrioritizedExperienceBuffer
from memory.uniform_memory import UniformExperienceBuffer
from utils.wrappers import LazyFrames


class DemoReplayBuffer(ReplayBufferAbstract):
    def __init__(
            self, prioritized: bool, capacity, demo_capacity, n_step, eps=0.01, alpha=0.6, beta0=0.4,
            betasteps=1000, gamma=0.99, ir_prob=0.2, n_policies=1, bonus_priority_demo=1.0,
            bonus_priority_agent=0.001, other_policy_factor=0.1):
        self.capacity = capacity
        self.demo_capacity = demo_capacity
        self.prioritized = prioritized
        memory_type = PrioritizedExperienceBuffer if prioritized else UniformExperienceBuffer

        self.agent_memory = memory_type(
            capacity=capacity, n_step=n_step, eps=eps, alpha=alpha, beta0=beta0, betasteps=betasteps,
            gamma=gamma, ir_prob=ir_prob, n_policies=n_policies, normalize_by_max=False)  # Handled in this cls)
        self.demo_memory = memory_type(
            capacity=demo_capacity, n_step=n_step, eps=eps, alpha=alpha, beta0=beta0, betasteps=betasteps,
            gamma=gamma, ir_prob=ir_prob, n_policies=n_policies, normalize_by_max=False)  # Handled in this cls)

        self.agent_samples = 0
        self.demo_samples = 0
        self.policy_idxs = None
        self.pretrain_phase = False
        self.current_policy = 0

        self.bonus_priority_demo = bonus_priority_demo
        self.bonus_priority_agent = bonus_priority_agent

    def set_pretrain_phase(self, pretraining: bool):
        self.pretrain_phase = pretraining

    def append(self, state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]] = None,
               action: Union[np.ndarray, int] = None, reward: int = 0, done: bool = False,
               next_state: Union[LazyFrames, Tuple[LazyFrames, np.ndarray]] = None,
               skip_step: int = 0, p_idx: int = 0, prob=None, expert: bool = False,
               **kwargs) -> None:
        memory = self.agent_memory
        if expert and (self.pretrain_phase and not self.demo_memory.is_full):
            memory = self.demo_memory
        memory.append(state=state, action=action, reward=reward, done=done, next_state=next_state,
                      skip_step=skip_step, p_idx=p_idx, prob=prob, expert=expert, **kwargs)

    def _sample_from_memory(self, nsample_agent, nsample_demo, p_idx):
        """Samples experiences from memory
        Args:
            nsample_agent (int): Number of RL transitions to sample
            nsample_demo (int): Number of demonstration transitions to sample
            p_idx (int): Index of priorities.
        """
        if nsample_demo > 0:
            sampled_demo, is_weight_demo = self.demo_memory.sample(
                nsample_demo, p_idx=p_idx)
            sampled_demo = sampled_demo.samples
        else:
            sampled_demo, is_weight_demo = [], []

        if nsample_agent > 0:
            sampled_agent, is_weight_agent = self.agent_memory.sample(
                nsample_agent, p_idx=p_idx)
            sampled_agent = sampled_agent.samples
        else:
            sampled_agent, is_weight_agent = [], []
        sampled_demo.extend(sampled_agent)
        sampled_demo = ExperienceSamples(sampled_demo)
        is_weight_demo = np.append(is_weight_demo, is_weight_agent)
        return sampled_demo, is_weight_demo

    def sample(self, n, demo_fraction=None, p_idx=0) -> Tuple[ExperienceSamples, np.ndarray]:
        """Sample `n` experiences from memory.
                Args:
                    n (int): Number of experiences to sample
                    demo_fraction (float): Fraction of experiences to come from demo memory. If set will override 50/50
                    p_idx (int): Index of priorities.
                """
        if demo_fraction == 1.0 or len(self.agent_memory) == 0:  # Demo only
            samples, is_weight = self._sample_from_memory(
                nsample_agent=0, nsample_demo=n, p_idx=p_idx)
            self.agent_samples = 0
            self.demo_samples = n
            return samples, is_weight
        if demo_fraction is not None:
            nsample_demo = int(demo_fraction*n)
            nsample_agent = n - nsample_demo
        else:
            priority_sums = self.agent_memory.sumtree[p_idx]
            priority_sums_demo = self.demo_memory.sumtree[p_idx]
            psum_agent = priority_sums.total_sum  # / (1 if self.pre_train_phase else len(self.memory))
            psum_demo = priority_sums_demo.total_sum  # / (1 if self.pre_train_phase else len(self.memory_demo))
            psample_agent = psum_agent / (psum_agent + psum_demo)

            # nsample_agent = int(n * (1.0 - demo_fraction))
            nsample_agent = np.random.binomial(n, psample_agent)
            # Increase agent samples with forget percentage -> leads to less demo samples, i.e. forgetting
            # nsample_agent += int((n - nsample_agent) * self.forget_percent)
            # If we don't have enough RL transitions yet, force more demos
            nsample_agent = min(nsample_agent, len(self.agent_memory))
            nsample_demo = n - nsample_agent

        samples, is_weight = self._sample_from_memory(
            nsample_agent, nsample_demo, p_idx=p_idx)
        self.agent_samples = nsample_agent
        self.demo_samples = nsample_demo
        self.policy_idxs = np.array(samples.policies)
        assert len(is_weight) == n
        return samples, is_weight

    def update_priorities(self, errors, p_idx=0):
        errors[:self.demo_samples] += self.bonus_priority_demo
        errors[-self.agent_samples:] += self.bonus_priority_agent
        errors = errors / errors.max()
        if self.agent_samples > 0:
            agent_errors = errors[-self.agent_samples:]
            self.agent_memory.update_priorities(agent_errors, p_idx=p_idx)
        if self.demo_samples > 0:
            demo_errors = errors[:self.demo_samples]
            self.demo_memory.update_priorities(demo_errors, p_idx=p_idx)

    def __len__(self):
        return len(self.demo_memory) + len(self.agent_memory)

    def stop_current_episode(self, env_id=0, p_idx=0):
        if self.pretrain_phase:
            memory = self.demo_memory if not self.demo_memory.is_full else self.agent_memory
        else:
            memory = self.agent_memory
        last_n_transitions = memory.last_n_transitions[env_id]
        memory.add_transition(last_n_transitions, done=True, p_idx=p_idx)

    def clear_transition_buffers(self):
        for memory in [self.demo_memory, self.agent_memory]:
            for val in memory.last_n_transitions.values():
                val.clear()
