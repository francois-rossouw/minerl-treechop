from typing import Union, List, Iterable, Tuple, Generator
from collections import deque

from tqdm import tqdm
import numpy as np
import torch
from torch import nn

from utils.minerl_wrappers import MyLazyFrames


def unsqueeze_obs(obs):
    if isinstance(obs, Tuple):
        return tuple(unsqueeze_obs(ob) for ob in obs)
    elif isinstance(obs, torch.Tensor):
        return obs.unsqueeze(0)


def totensor(obs: Union[List[Union[np.ndarray, MyLazyFrames]], np.ndarray, MyLazyFrames],
             device) -> torch.Tensor:
    """Converts input to tensor. Input can be:
    LazyFrames, numpy arrays, List of these types, Tuple of the aforementioned.
    Output will be ready for use in PyTorch NN:
    Tensor, Tensor, Stacked Tensor, Tuple of stacked tensors
    :param obs: Data to be made to tensor
    :param device: Device of network. If CUDA, speeds up conversion to float + division
    :return:
    """
    if isinstance(obs, MyLazyFrames):
        obs = np.asarray(obs)
    if isinstance(obs, np.ndarray):
        if obs.ndim == 3:
            return torch.from_numpy(obs).to(device).float().div(255)
        elif obs.ndim == 1:
            print(obs)
            return torch.tensor(obs, dtype=torch.float)
    if isinstance(obs, (List, Generator)):
        obs = np.array([np.array(ob) for ob in obs])
        return torch.from_numpy(obs).to(device).float().div(255)


def soft_update(online: nn.Module, target: nn.Module, tau: float = 1e-2):
    """Soft-update: target = tau*local + (1-tau)*target."""
    for t_param, o_param in zip(target.parameters(), online.parameters()):
        t_param.data.copy_(tau * o_param.data + (1.0 - tau) * t_param.data)


def print_b(text):
    """Print function for when progressbar is running
    :param text: Text to print
    :return: None
    """
    tqdm.write(str(text))


def seed_things(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StackSet:
    def __init__(self, stack: Union[List, Iterable] = None, maxlen: int = None, circular: bool = True):
        """Stack FIFO style with unique values (set). Also supports a maximum length
        :param stack: Initializing list
        :param maxlen: Maxlength of stack, will remove from index 0 if is_full
        :param circular: Append value from pop back to top of stack
        """
        if stack is not None:
            self.stack = deque(list(stack), maxlen=maxlen)
        else:
            self.stack = deque([], maxlen=maxlen)
        self.circular = circular
        self.maxlen = maxlen

    def push(self, v) -> None:
        """
        Push value to stack. Deletes occurances of value in stack.
        :param v: Value to push
        :return: None
        """
        stack = self.stack
        if v in stack:
            del stack[stack.index(v)]
        stack.append(v)

    def pop(self, idx=0, val=None) -> object:
        """Pop index from stack, defaults to start of stack. If circular is set, will push the value back to the
        top of the stack.
        :param idx: Index to pop (default 0)
        :param val: Allows user to pop element by value
        :return: Value popped from stack
        """
        stack = self.stack
        if val is not None:
            assert idx == 0
            idx = stack.index(val)
        v = stack[idx]
        if self.circular and idx == 0:  # Speed shortcut with deque
            stack.rotate(-1)
            return v
        del stack[idx]
        if self.circular:
            self.push(v)  # Move back to end
        return v


if __name__ == '__main__':
    s = StackSet(range(20))
    ir_prob = 0.2
    for i in range(100):
        if np.random.rand() < ir_prob:
            r_idx = np.random.choice(list(s.stack))
            s.pop(val=r_idx)
        else:
            s.pop()
        print(s.stack)
