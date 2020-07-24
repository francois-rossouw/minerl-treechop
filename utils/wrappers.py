"""
Default atari wrappers
"""
import os
import numpy as np
import gym
import minerl
import cv2
from collections import deque
from gym import spaces
from typing import List, Union, Dict
from collections import OrderedDict
import copy
import math
from gym.wrappers.monitor import Monitor


class CamDiscrete(gym.spaces.Discrete):
    """
    Wrapper for when continuous value is discretized but zero is at a different index
    """
    def __init__(self, n):
        super(CamDiscrete, self).__init__(n)
        assert n % 2 != 0
        self.zero_idx = math.floor(n/2)

    def no_op(self):
        return self.zero_idx


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque([], maxlen=2)
        self._skip = skip
        self._obs_isdict = isinstance(self.observation_space, spaces.Dict)  # Assume MineRL env

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        info = None
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if not self._obs_isdict:
            obs = np.maximum(
                self._obs_buffer[0],
                self._obs_buffer[1]
            )
        else:
            obs['pov'] = np.maximum(
                self._obs_buffer[0]['pov'],
                self._obs_buffer[1]['pov']
            )
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class FrameSkip(gym.Wrapper):
    """
    Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped _frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        # Do not repeat these actions
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        # if self._out is None:
        #     self._out = np.concatenate(self._frames, axis=-1)
        #     self._frames = None
        # return self._out
        return np.concatenate(self._frames, axis=-1)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class ContToDisc(gym.ActionWrapper):
    def __init__(self, env, bins: List[np.ndarray] = None, n_bins=None, clip_val: float = None):
        super().__init__(env)
        self._wrapping_action_space: Union[spaces.Dict, spaces.Box] = env.action_space
        self._wrap_action_space_isdict = isinstance(self._wrapping_action_space, spaces.Dict)
        self._sample_act = self._wrapping_action_space.sample()
        box_instances = []
        if isinstance(self._wrapping_action_space, spaces.Dict):
            # Dict instance with boxes inside
            box_inst_fnd = False
            for space in self._wrapping_action_space.spaces.values():
                if isinstance(space, spaces.Box):
                    box_inst_fnd = True
                    box_instances.append(space)
            assert box_inst_fnd
        else:
            # Only one box instance
            assert isinstance(self._wrapping_action_space, spaces.Box)
            box_instances.append(self._wrapping_action_space)

        if bins is None:  # Need to automate bins
            assert n_bins is not None
            self.bins = ContToDisc._generate_bins(box_instances, n_bins, clip_val)
            self.n_bins = n_bins
        else:
            self.bins = bins
            self.n_bins = len(bins[0])

        self.action_space, self.bins = ContToDisc.wrap_action_space(bins, env.action_space)

    @staticmethod
    def wrap_action_space(bins, action_space: Union[spaces.Box, spaces.Dict]):
        wrap_action_space_isdict = isinstance(action_space, spaces.Dict)
        is_minerl_env = isinstance(action_space, minerl.env.spaces.Dict)
        if len(bins) == 1 and not wrap_action_space_isdict and \
                np.prod(action_space.shape) == 1:
            return spaces.Discrete(len(bins[0][0])), bins
        else:
            wrapping_action_space = copy.copy(action_space)
            action_space = OrderedDict()
            disc_acts = OrderedDict()
            bins = bins[0] if not wrap_action_space_isdict else bins
            ContToDisc.make_action_space(bins, action_space, 'action', wrapping_action_space, disc_acts)
            dict_space = spaces.Dict if not is_minerl_env else minerl.env.spaces.Dict
            return dict_space(action_space), disc_acts

    @staticmethod
    def make_action_space(bins: Union[List[np.ndarray], np.ndarray], action_space: OrderedDict, action_name: str,
                          space: Union[spaces.Dict, spaces.Box], disc_acts: OrderedDict):
        is_minerl_env = isinstance(space, (minerl.env.spaces.Dict, minerl.env.spaces.Box))
        if isinstance(space, spaces.Dict):
            # Got Dict action space
            bin_iter = iter(bins)
            for idx, (key, value) in enumerate(space.spaces.items()):
                if isinstance(value, spaces.Box):
                    ContToDisc.make_action_space(next(bin_iter), action_space, key, value, disc_acts)
                else:
                    action_space[key] = value
        elif isinstance(space, spaces.Box):
            combinations = shape_combinations(space.shape)
            for c in combinations:
                b = bins[0] if isinstance(bins[0], list) else bins
                key = [action_name, '_']
                key.extend([str(dim) for dim in c])
                key = ''.join(key)
                discrete_space = spaces.Discrete if not is_minerl_env else CamDiscrete
                action_space[key] = discrete_space(len(b))
                disc_acts[key] = b

    @staticmethod
    def _generate_bins(box_instances: List[gym.spaces.Box], n_bins: int, clip_val: float):
        bins = []
        for box in box_instances:
            low, high = box.low, box.high
            if clip_val is not None:
                low, high = low.clip(-clip_val), high.clip(max=clip_val)
            bins.append(
                np.linspace(start=low, stop=high, num=n_bins, endpoint=True).T
            )
        return bins

    def action(self, action: Union[Dict, OrderedDict, int]):
        if isinstance(action, Dict) or isinstance(action, OrderedDict):
            action_copy = copy.copy(action)
            boxes = {key: action_copy.pop(key) for key in action if key in self.bins}
            if self._wrap_action_space_isdict:
                self._sample_act.update(action_copy)
            for key, val in boxes.items():
                key_prefix, idx = key.split('_')
                indices = tuple(map(int, idx))
                bin_indices = indices + (val,)
                if self._wrap_action_space_isdict:
                    self._sample_act[key_prefix][indices] = self.bins[key][val]
                else:
                    self._sample_act[indices] = self.bins[key][bin_indices]
            return self._sample_act
        elif isinstance(action, int):
            act = np.array([self.bins[0][0][action]])
            return act
        else:
            raise TypeError(f"Only accepts int, dict or OrderedDict types, got {type(action)}")


def make_atari(env_id):
    """
    Create a wrapped atari envrionment
    :param env_id: (str) the environment ID
    :return: (Gym Environment) the wrapped atari environment
    """
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # env = FrameSkip(env, skip=4)
    return env


def wrap_deepmind(env, clip_rewards=True, frame_stack=True):
    """Configure environment for DeepMind-style Atari.
    """
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_env(env_id):
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    return env


def wrap_monitor(env, log_dir):
    from gym import wrappers
    env = wrappers.Monitor(
        env, log_dir, video_callable=lambda x: True)
    return env


def shape_combinations(shape):
    shape = tuple(list(range(dim)) if idx % 2 == 0 else list(reversed(range(dim))) for idx, dim in enumerate(shape))
    n_dims = len(shape)
    mesh = np.array(np.meshgrid(*shape))
    return np.unique(mesh.T.reshape(-1, n_dims), axis=0)


if __name__ == '__main__':
    import minerl
    env = gym.make('LunarLanderContinuous-v2')
    env = ContToDisc(env, n_bins=7, clip_val=5)
    env.reset()
    # env.step(OrderedDict([
    #     ('attack', 0),
    #     ('back', 1),
    #     ('camera_0', 3),
    #     ('camera_1', 4),
    #     ('forward', 0),
    #     ('jump', 1),
    #     ('left', 0),
    #     ('right', 1),
    #     ('sneak', 0),
    #     ('sprint', 1)
    # ]))
    env.step(OrderedDict([
        ('action_0', 0),
        ('action_1', 1),
    ]))
