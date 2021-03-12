import time
from statistics import mean
from dataclasses import dataclass
import numpy as np
import torch
import wandb

from utils.write_to_excel import DataWriter


class Timer:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = time.time()

    @property
    def elapsed(self):
        assert self.start_time is not None, 'Timer has not been started yet!'
        if self.stop_time is None:
            return time.time() - self.start_time
        else:
            return self.stop_time - self.start_time

    def subtract_duration(self, duration):
        assert self.start_time is not None, 'Timer has not been started yet!'
        self.start_time -= duration

    def stop(self):
        self.stop_time = time.time()


@dataclass
class Logger:
    """Class to track stats listed below. Also some functions to extend the use case."""
    use_wandb: bool
    step: int = 0
    episode: int = 1
    max_score: float = -np.inf
    total_steps: int = 0
    av_length: int = 100
    best_running_score: float = -np.inf
    loss: list = None
    epsilon: float = 1.0
    epsilon_start: float = None
    rewards: list = None
    ep_rewards: list = None
    q_vals: list = None
    timer: Timer = None
    ep_step_cnt: int = 0
    expert_percent: float = 0.0
    dw: DataWriter = None

    def __post_init__(self):
        """Gets called after the default init of a dataclass"""
        from utils.utilities import print_b
        self.print_b = print_b
        self.rewards = []
        self.ep_rewards = []
        self.loss = []
        self.timer = Timer()
        self.epsilon_start = self.epsilon

    def create_datawriter(self, args):
        self.dw = DataWriter(args)

    def init_tracker(self, **kwargs):
        """
        Initializes instance to values given in kwargs. Starts timer and accounts for continued runs if given a
        'duration' key.
        :param kwargs:
        :return: None
        """
        for k, v in kwargs.items():
            if k in self.__dict__.keys() and k is not 'timer':
                setattr(self, k, v)
            elif k is not 'timer':
                print(f'There is no {k} key in the StatTracker class.')

        self.start_timer()
        if 'timer' in kwargs:
            self.timer.subtract_duration(kwargs['timer'].elapsed)

    def start_timer(self):
        self.timer.start()

    def stop_timer(self):
        self.timer.stop()

    def get_time_str(self):
        """Get time in a readable format. (00h 00m 00s)"""
        duration = self.timer.elapsed
        return '{:02d}h {:02d}m {:02d}s'.format(
            int(duration / 3600), int((duration % 3600) / 60), int(duration % 60)
        )

    def append_reward(self, reward):
        self.rewards.append(reward)

    def add_loss(self, loss):
        assert not isinstance(loss, tuple)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(loss, np.ndarray):
            loss = np.mean(loss)
        if loss != 0.0:
            self.loss.append(float(loss))

    def finish_episode(self, verbose):
        """
        Update parameters and print current progress
        :param verbose: Print current progress
        :return: None
        """
        dur, sum_reward, expert_percent = self._get_print_data()
        if self.use_wandb:
            self._log_wandb(sum_reward)
        if verbose:
            self.print_progress(dur, sum_reward, expert_percent=expert_percent)
        self.rewards = []
        self.loss = []

    def _log_wandb(self, sum_reward):
        wandb.log({'reward': int(sum_reward)}, step=self.step)
        wandb.log({'loss': self.mean_loss}, step=self.step)
        wandb.log({'epsilon': self.epsilon}, step=self.step)
        wandb.log({'median_reward': self.av_reward}, step=self.step)

    def _get_print_data(self):
        sum_reward = sum(self.rewards)
        self.ep_step_cnt += 1 if self.ep_step_cnt == 0 else 0
        expert_percent = f"{(self.expert_percent / self.ep_step_cnt) * 100:.2f} %"
        self.ep_step_cnt = 0
        self.expert_percent = 0
        self.ep_rewards.append(sum_reward)
        if sum_reward > self.max_score:
            if self.use_wandb:
                wandb.run.summary["max_score"] = int(sum_reward)
            self.max_score = sum_reward
        if self.av_reward > self.best_running_score:
            if self.use_wandb:
                wandb.run.summary["max_median_score"] = int(self.av_reward)
            self.best_running_score = self.av_reward

        past30reward = self.ep_rewards if len(self.ep_rewards) < self.av_length else self.ep_rewards[-self.av_length:]

        dur = self.get_time_str()
        if self.dw is not None:
            self.dw.write_episode_data(self.episode, dur, past30reward, expert_percent,
                                       f"{self.mean_loss:.4f}", total_steps=self.step)
        return dur, int(sum_reward), expert_percent

    def add_expert_percentage(self, expert_percent):
        self.ep_step_cnt += 1
        self.expert_percent += expert_percent

    def print_progress(self, duration, sum_reward, tqdm_print=True, expert_percent="0.0 %"):
        duration_str = ''.join([
            f'Ep: {self.episode}',
            f',  {duration}:',
            f'    reward: {int(sum_reward)}',
            f'    experts: {expert_percent}  ',
            f'    epsilon: {self.epsilon:.3f}'
        ])
        str3 = 'loss: {:.4f}, mean reward: {:.1f}, step: {},    best_reward: {},   best_running_reward: {:.3f}'.format(
            self.mean_loss, self.av_reward, self.step+1, int(self.max_score), self.best_running_score
        )
        to_print = '\t'.join([duration_str, str3])

        if tqdm_print:
            self.print_b(to_print)
        else:
            print(to_print)

    @property
    def mean_loss(self):
        if len(self.loss) == 1:
            loss = self.loss[0]
        elif len(self.loss) > 1:
            loss = mean(self.loss)
        else:
            loss = 0.0
        return loss

    @property
    def av_reward(self):
        if len(self.ep_rewards) < self.av_length:
            # return median(self.ep_rewards)
            return 0.0
        return mean(self.ep_rewards[-self.av_length:])

    @property
    def total_reward(self):
        return sum(self.rewards) if self.rewards else 0

    def update_epsilon(self, subtracts, final_eps, epsilon_steps):
        curr_step = self.step - subtracts + 1
        progress = max(1.0, epsilon_steps / curr_step)
        self.epsilon = self.epsilon_start - (self.epsilon_start - final_eps) / progress
