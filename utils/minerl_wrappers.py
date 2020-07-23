import os
import math
import copy
from typing import Tuple, Union, List, Dict
from collections import deque, OrderedDict, Counter
from itertools import repeat

import cv2
import numpy as np
import gym
from gym.wrappers.monitor import Monitor
import minerl
from minerl.herobraine.hero import spaces

from utils.gen_args import Arguments
from utils.wrappers import LazyFrames


class CamDiscrete(gym.spaces.Discrete):
    def __init__(self, n):
        super(CamDiscrete, self).__init__(n)
        self.zero_idx = math.floor(n/2)

    def sample(self, *args, **kwargs):
        return super().sample()

    def no_op(self, *args, **kwargs):
        return self.zero_idx


class TupleSpace(gym.Space):
    def __init__(self, spaces: Tuple[gym.Space, gym.Space]):
        self.spaces = spaces
        for space in spaces:
            assert isinstance(space, gym.Space), "Elements of the tuple must be instances of gym.Space"
        super(TupleSpace, self).__init__(spaces[0].shape, None)
        assert len(spaces) == 2
        self.other_shape = spaces[1].shape

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def __repr__(self):
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)

    def __eq__(self, other):
        return isinstance(other, TupleSpace) and self.spaces == other.spaces


class PoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = PoVWrapper.pov_obs_wrapper(env.observation_space)

    @staticmethod
    def pov_obs_wrapper(observation_space: spaces.Dict):
        return observation_space.spaces['pov']

    def observation(self, observation):
        return observation['pov']


class PoVInvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = PoVWrapper.pov_obs_wrapper(env.observation_space)

    @staticmethod
    def pov_obs_wrapper(observation_space: spaces.Dict):
        obs_spaces = observation_space.spaces
        return TupleSpace((obs_spaces['pov'], obs_spaces['inventory']))

    def observation(self, obs):
        inv = np.stack(list(obs['inventory'].values()))
        mainhand = obs['equipped_items']['mainhand']['type']
        inv = np.concatenate(([mainhand if not isinstance(mainhand, str) else 8], inv))
        return obs['pov'], inv


class ContToDiscActions(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.
    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.
    """
    def __init__(self, env, bins: list):
        super().__init__(env)
        self.bins = bins
        self.ex_action = self.env.action_space.no_op()
        self.action_space = ContToDiscActions.wrap_action_space(self.env.action_space, bins)

    @staticmethod
    def wrap_action_space(action_space: spaces.Dict, bins) -> spaces.Dict:
        wrapping_action_space = copy.deepcopy(action_space.spaces)
        discrete = OrderedDict([
            ('camera_0', CamDiscrete(len(bins))),
            ('camera_1', CamDiscrete(len(bins)))
        ])
        wrapping_action_space.pop('camera', None)
        for key, value in discrete.items():
            wrapping_action_space[key] = value
        return spaces.Dict(wrapping_action_space)

    def action(self, action):
        return self.fix_action(action)

    def fix_action(self, action):
        pitch = action.pop('camera_0', None)
        yaw = action.pop('camera_1', None)
        self.ex_action.update(action)
        self.ex_action['camera'][0] = self.bins[pitch]
        self.ex_action['camera'][1] = self.bins[yaw]
        return self.ex_action


class ExcludeActions(gym.ActionWrapper):
    """
    Remove actions from action space assuming a dict action space from MineRL
    """
    def __init__(self, env, excluded_actions: list = None):
        super().__init__(env)
        self.ex_action = self.env.action_space.no_op()
        self.action_space = ExcludeActions.wrap_action_space(self.env.action_space, excluded_actions)

    @staticmethod
    def wrap_action_space(action_space: spaces.Dict, excluded_actions: List[str]):
        wrapping_action_space = copy.deepcopy(action_space.spaces)

        if excluded_actions:
            for key in excluded_actions:
                wrapping_action_space.pop(key)

        return spaces.Dict(wrapping_action_space)

    def action(self, action):
        self.ex_action.update(action)
        return self.ex_action


class AlwaysActions(gym.ActionWrapper):
    """Convert MineRL env's `Dict` action space as a serial discrete action space.
    The term "serial" means that this wrapper can only push one key at each step.
    "attack" action will be alwarys triggered.
    """
    def __init__(self, env, always_actions: List[str] = None):
        super().__init__(env)
        self.ex_action = self.env.action_space.no_op()
        for key in self.ex_action:
            if key in always_actions:
                self.ex_action[key] = 1
        self.action_space = AlwaysActions.wrap_action_space(self.env.action_space, always_actions)

    @staticmethod
    def wrap_action_space(action_space: spaces.Dict, always_actions: List[str]):
        wrapping_action_space = copy.deepcopy(action_space.spaces)

        if always_actions:
            for key in always_actions:
                wrapping_action_space.pop(key)

        return spaces.Dict(wrapping_action_space)

    def action(self, action):
        self.ex_action.update(action)
        return self.ex_action


class MineRLFrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped _frames.
    """
    def __init__(self, env, craft_acts, skip=4):
        super().__init__(env)
        # Do not repeat these actions
        self._skip = skip
        self._craft_acts_noop = OrderedDict([
            (key, 0) for key in craft_acts
        ])

    def step(self, action: OrderedDict):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
            if set(action).intersection(self._craft_acts_noop):  # Craft actions present
                action.update(self._craft_acts_noop)
        return obs, total_reward, done, info


class MineRLFrameStack(gym.Wrapper):
    def __init__(self, env, k, obtain_env: bool = True):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.invs = deque([], maxlen=k)
        pov_observation_space = self.observation_space.spaces['pov']
        shp = pov_observation_space.shape
        self._out_shape = list((shp[-1] * k,) + shp[:-1])
        self.observation_space = MineRLFrameStack.wrap_obs_space(env.observation_space, k=k)

    @staticmethod
    def wrap_obs_space(observation_space: spaces.Dict, k: int):
        pov_observation_space = observation_space.spaces['pov']
        shp = pov_observation_space.shape
        observation_space.spaces['pov'] = spaces.Box(
            low=0, high=255,
            shape=((shp[-1] * k,) + shp[:-1]),
            dtype=pov_observation_space.dtype)
        return observation_space

    def reset(self, **kwargs):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob['pov'].transpose(2, 0, 1))
        ob['pov'] = self._get_ob()
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob['pov'].transpose(2, 0, 1))
        ob['pov'] = self._get_ob()
        return ob, reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return MyLazyFrames(list(self.frames), self._out_shape)


class OneMovementAction(gym.ActionWrapper):
    """
    Split action space into movement & crafting actions. Similar to Hierarchical Deep Q-Network from Imperfect
    Demonstrations in Minecraft paper at https://arxiv.org/pdf/1912.08664.pdf
    """
    def __init__(self, env, craft_keys: List[str]):
        super().__init__(env)
        self.ex_action = self.env.action_space.no_op()
        self.wrapping_action_space = env.action_space
        self.action_space = OneMovementAction.wrap_action_space(env.action_space, craft_keys)
        self._action_combinations = OrderedDict([
            (0, self._get_combined_action()),
            (1, self._get_combined_action(camera=('camera_0', 2))),
            (2, self._get_combined_action(camera=('camera_0', 0))),
            (3, self._get_combined_action(camera=('camera_1', 2))),
            (4, self._get_combined_action(camera=('camera_1', 0))),
            (5, self._get_combined_action(actions=['forward'])),
            (6, self._get_combined_action(actions=['forward', 'jump'])),
            (7, self._get_combined_action(actions=['left'])),
            (8, self._get_combined_action(actions=['right'])),
            (9, self._get_combined_action(actions=['back'])),
            (10, self._get_combined_action(actions=['forward', 'sprint'])),
            (11, self._get_combined_action(actions=['jump'])),
        ])

    @staticmethod
    def wrap_action_space(action_space: spaces.Dict, craft_keys: List[str]):
        action_space = copy.deepcopy(action_space.spaces)
        new_action_space = OrderedDict([
            (key, action_space.pop(key)) for key in craft_keys if key in action_space
        ])
        movement_actions = spaces.Discrete(12)
        new_action_space['movement'] = movement_actions
        # Move to top
        new_action_space.move_to_end('movement', last=False)
        return spaces.Dict(new_action_space)

    def _get_combined_action(self, actions: List[str] = None, camera: Tuple[str, int] = None):
        action = self.wrapping_action_space.no_op()
        if actions is not None:
            for act in actions:
                action[act] = 1

        if camera is not None:
            axis, val = camera
            action[axis] = val
        return action

    def action(self, action):
        movement_action = self._action_combinations[action.pop('movement')]
        self.ex_action.update(movement_action)
        self.ex_action.update(action)
        return self.ex_action


class DictClone(gym.ActionWrapper):
    """
    Add this after all other action wrappers to ensure action dicts are not modified in replay memory.
    Makes copying actions unnecessary. Also ensures specific order to actions.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Dict), \
            f"Only applicable for dict action spaces. Got {type(env.action_space)}"
        self.action_space = copy.copy(env.action_space)
        self.ex_action = env.action_space.no_op()

    def action(self, action: Union[Dict, OrderedDict]):
        ex_action = self.action_space.no_op()
        ex_action.update(action)
        return ex_action


class MyLazyFrames(LazyFrames):
    def __init__(self, frames, out_shape: list):
        super(MyLazyFrames, self).__init__(frames)
        """Small adaptation for nearly 4x speedup. Need to transpose frames before this."""
        self._out_shape = out_shape

    def _force(self):
        return np.array(self._frames).reshape(self._out_shape)  # .transpose((0, 3, 1, 2))

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self.frame(i)

    def count(self):
        return len(self)

    def frame(self, i):
        return self._force()[i]


class LazyFramesFlipped(MyLazyFrames):
    def _force(self):
        out = super()._force()
        return np.flip(out, axis=2).copy()


class LogRewardData(gym.Wrapper):
    """Take first return value.
    minerl's `env.reset()` returns tuple of `(obs, info)`
    but existing agents implementations expect `reset()` returns `obs` only.
    """
    def __init__(self, env, episode=0):
        super().__init__(env)
        self.filename = '/'.join([os.getcwd(), 'logs', 'rewards.txt'])
        open(self.filename, 'w').close()  # Clear file contents
        self.reward_hist = []
        self.reward_keys = [
            "log", "planks", "stick", "crafting_table", "wooden_pickaxe", "cobblestone",
            "furnace", "stone_pickaxe", "iron_ore", "iron_ingot", "iron_pickaxe"]
        self.reward_types = {
            "log": 1,
            "planks": 2,
            "stick": 4,
            "crafting_table": 4,
            "wooden_pickaxe": 8,
            "cobblestone": 16,
            "furnace": 32,
            "stone_pickaxe": 32,
            "iron_ore": 64,
            "iron_ingot": 128,
            "iron_pickaxe": 256,
            "diamond": 1024
        }
        self.total_reward = 0
        self.non_dense_reward = 0
        self.prev_inv = None
        self.episode = episode

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        if reward > 0:
            self.total_reward += reward
            for key, val in ob['inventory'].items():
                val_diff = val - self.prev_inv[key]
                if val_diff > 0 and key in self.reward_keys:
                    if key not in self.reward_hist:
                        self.non_dense_reward += self.reward_types[key]
                    for _ in range(val_diff):
                        self.reward_hist.append(key)
                    break
        self.prev_inv = copy.deepcopy(ob['inventory'])
        return ob, reward, done, info

    def reset(self, **kwargs):
        if self.reward_hist:  # Ensure there is data to write
            self._write_log(self.episode)
            self.total_reward = 0
            self.non_dense_reward = 0
            self.reward_hist = []
        self.episode += 1
        obs = self.env.reset()
        self.prev_inv = copy.copy(obs['inventory'])
        return obs

    def _write_log(self, episode):
        count_str = ', '.join([
            f'{amount} {key}' for key, amount in Counter(self.reward_hist).items()
        ])
        with open(self.filename, 'a') as file:
            file.write(f'Episode: {episode};  reward: {self.total_reward}\n'
                       f'Non-dense reward: {self.non_dense_reward}\n'
                       f'Reward consisting of: {count_str}\n')


class ResizeFrame(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env, width=64, height=64):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space, self._do_resize = ResizeFrame.wrap_obs_space(
            env.observation_space, width=width, height=height)

    @staticmethod
    def wrap_obs_space(observation_space: spaces.Dict, width=64, height=64):
        observation_space = copy.deepcopy(observation_space)
        do_resize = False
        if observation_space.spaces['pov'].shape[0] > width:
            do_resize = True
            observation_space.spaces['pov'].shape = (
                width, height, observation_space.spaces['pov'].shape[-1]
            )
        return observation_space, do_resize

    def observation(self, observation):
        if self._do_resize:
            observation['pov'] = cv2.resize(
                observation['pov'], (self._width, self._height), interpolation=cv2.INTER_LINEAR
            )
        return observation


def make_minerl_env(env_id, args: Arguments):
    env = gym.make(env_id)
    # wrap env: observation...
    # if args.test or args.monitor:
    if args.monitor:
        env = Monitor(
            env, os.path.join(args.outdir, env.spec.id, 'monitor'),
            mode='evaluation' if args.test else 'training', video_callable=lambda episode_id: True
        )
        env = ResizeFrame(env)  # For monitoring larger _frames.

    # env = MaxAndSkipEnv(env, skip=4)
    env = MineRLFrameSkip(env, craft_acts=args.crafting_actions, skip=4)
    # if args.frame_stack > 1:
    env = MineRLFrameStack(env, k=4)
    if 'inventory' in env.observation_space.spaces:
        env = PoVInvWrapper(env)
    else:
        env = PoVWrapper(env)

    # wrap env: action...
    env = ExcludeActions(env, excluded_actions=args.exclude_actions)
    env = ContToDiscActions(env, args.bins)
    if not args.action_branching:
        env = AlwaysActions(env, always_actions=['attack'])
        env = OneMovementAction(env, args.crafting_actions)
    env = DictClone(env)  # Ensures we do not modify stored action dicts
    return env


def wrap_minerl_obs_space(observation_space: spaces.Dict) -> Union[spaces.Box, TupleSpace]:
    observation_space = copy.deepcopy(observation_space)
    if observation_space.spaces['pov'].shape[1] > 64:
        observation_space, _ = ResizeFrame.wrap_obs_space(observation_space)
    observation_space = MineRLFrameStack.wrap_obs_space(observation_space, k=4)
    if 'inventory' in observation_space.spaces:
        observation_space = PoVInvWrapper.pov_obs_wrapper(observation_space)
    else:
        observation_space = PoVWrapper.pov_obs_wrapper(observation_space)
    return observation_space


def wrap_minerl_action_space(args: Arguments, action_space: spaces.Dict):
    action_space = copy.deepcopy(action_space)
    action_space = ExcludeActions.wrap_action_space(action_space, args.exclude_actions)

    action_space = ContToDiscActions.wrap_action_space(action_space, args.bins)
    action_space.spaces.move_to_end('camera_0')
    action_space.spaces.move_to_end('camera_1')
    pretrain_act_space = copy.deepcopy(action_space)
    if not args.action_branching:
        action_space = AlwaysActions.wrap_action_space(action_space, ['attack'])
        pretrain_act_space = AlwaysActions.wrap_action_space(pretrain_act_space, ['attack'])
        action_space = OneMovementAction.wrap_action_space(action_space, craft_keys=args.crafting_actions)
    return action_space, pretrain_act_space


if __name__ == '__main__':
    args = Arguments(underscores_to_dashes=False).parse_args(known_only=False)
    env_spec = gym.envs.registry.spec(args.env_name)
    observation_space = env_spec._kwargs['observation_space']  # Private variable contains what we need
    action_space = env_spec._kwargs['action_space']  # Private variable contains what we need
    print(wrap_minerl_obs_space(observation_space))
    print(wrap_minerl_action_space(args, action_space))
