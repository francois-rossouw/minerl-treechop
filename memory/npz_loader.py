import os

import gym
import numpy as np
from collections import deque
import copy
from typing import Union, Dict, Tuple, List
from minerl.herobraine.hero import spaces
from collections import Counter

from memory.demo_memory import DemoReplayBuffer
from memory.dataset_loader import ExpertDataset
from utils.minerl_wrappers import MyLazyFrames
from utils.gen_args import Arguments

REW_RANGE_DICT = {
    'Acrobot-v1': (-500, 0),
    'CartPole-v0': (-200, 200)
}


def load_numpy_data(game_name: str) -> Tuple[np.ndarray, ...]:
    min_rew, max_rew = REW_RANGE_DICT[game_name]
    demonstrations_dir = os.path.join("demonstrations")
    data = dict()
    for root, folder, files in os.walk(demonstrations_dir):
        for filename in files:
            if game_name.lower() in filename.lower():
                expert_data = dict(np.load(os.path.join(demonstrations_dir, filename), allow_pickle=True))
                expert_data['n_obs'] = np.array([x[1:].copy() for x in expert_data['obs']])
                expert_data['obs'] = np.array([x[:-1].copy() for x in expert_data['obs']])
                expert_data['expert_scale'] = copy.deepcopy(expert_data['rew'])

                for key, values in expert_data.items():
                    v_list = []
                    for idx, val in enumerate(values.tolist()):
                        if key == 'expert_scale':
                            scale = (sum(val) - min_rew) / abs(max_rew - min_rew)
                            val[:] = [scale]
                        v_list.extend(val)
                    np_values = np.array(v_list)
                    if len(np_values) > 0:
                        if key == 'acs':
                            np_values = np_values.astype(np.uint8)
                        if key in data:
                            data[key] = np.concatenate((data[key], np_values.squeeze()), axis=0)
                        else:
                            data[key] = np_values.squeeze()
    assert data['done'].shape[0] == data['obs'].shape[0] == data['n_obs'].shape[0], f"{data['done'].shape[0]} vs {data['obs'].shape[0]} vs {data['n_obs'].shape[0]}"
    return tuple(data.values())


def read_npz(args: Arguments, fill_size, memory: DemoReplayBuffer):
    observation_space = get_obs_space(args)
    if isinstance(observation_space, gym.spaces.Box):
        out_shape = observation_space.shape
    elif isinstance(observation_space, gym.spaces.Discrete):
        out_shape = [observation_space.n]
    else:
        raise NotImplementedError(f"No support for observation spaces other than Discrete or Box. Got {type(observation_space)}")
    frame_hist = deque([], maxlen=args.frame_stack)
    n_frame_hist = deque([], maxlen=args.frame_stack)
    states, actions, rewards, dones, n_states, expert_scales = load_numpy_data(args.env_name)
    for idx, (state, action, reward, done, n_state, expert_scale) in enumerate(zip(states, actions, rewards, dones, n_states, expert_scales)):
        if idx >= fill_size:
            return
        frame_hist.append(state)
        n_frame_hist.append(n_state)
        while len(frame_hist) < args.frame_stack:
            frame_hist.append(state)
            n_frame_hist.append(n_state)

        memory.append(
            state=MyLazyFrames(list(frame_hist), out_shape),
            action=action,
            reward=reward,
            done=done,
            next_state=MyLazyFrames(list(n_frame_hist), out_shape),
            expert=True,
            expert_scale=expert_scale
        )
        if done:
            memory.stop_current_episode()
            frame_hist.clear()
            n_frame_hist.clear()
    print(f"{len(memory)} transitions loaded from npz files.")


def get_obs_space(args: Arguments):
    sample_env = gym.make(args.env_name)
    observation_space = sample_env.observation_space
    sample_env.close()
    return observation_space


if __name__ == '__main__':
    states, actions, rewards, dones, n_states, expert_scales = load_numpy_data("Acrobot-v1")
