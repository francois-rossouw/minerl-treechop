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

MIN_REW_DICT = {
    'Acrobot-v1': -120,
    'CartPole-v0': 150
}


def load_numpy_data(game_name: str) -> Tuple[np.ndarray, ...]:
    min_rew = MIN_REW_DICT[game_name]
    demonstrations_dir = os.path.join("demonstrations")
    data = dict()
    for root, folder, files in os.walk(demonstrations_dir):
        for filename in files:
            if game_name.lower() in filename.lower():
                expert_data = dict(np.load(os.path.join(demonstrations_dir, filename), allow_pickle=True))
                bad_idxs = []
                for i, r in enumerate(expert_data['rew']):
                    if sum(r) < min_rew:
                        bad_idxs.append(i)
                for key, values in expert_data.items():
                    v_list = []
                    for idx, val in enumerate(values.tolist()):
                        if idx in bad_idxs:
                            continue
                        if key == 'obs':
                            v_list.extend(val[:-1])
                        else:
                            v_list.extend(val)
                    np_values = np.array(v_list)
                    if len(np_values) > 0:
                        if key == 'acs':
                            np_values = np_values.astype(np.uint8)
                        if key in data:
                            data[key] = np.concatenate((data[key], np_values.squeeze()), axis=0)
                        else:
                            data[key] = np_values.squeeze()
    assert data['done'].shape[0] == data['obs'].shape[0], f"{data['done'].shape[0]} vs {data['obs'].shape[0]}"
    return data['obs'], data['acs'], data['rew'], data['done']


def read_npz(args: Arguments, fill_size, memory: DemoReplayBuffer):
    frame_hist = deque([], maxlen=args.frame_stack)
    states, actions, rewards, dones = load_numpy_data(args.env_name)
    for idx, (state, action, reward, done) in enumerate(zip(states, actions, rewards, dones)):
        if idx >= fill_size:
            return
        frame_hist.append(state)
        if len(frame_hist) < args.frame_stack:
            continue

        memory.append(
            state=MyLazyFrames(list(state)),
            action=action,
            reward=reward,
            done=done,
            expert=True
        )
        if done:
            memory.stop_current_episode()
    print(f"{len(memory)} transitions loaded from npz files.")


if __name__ == '__main__':
    load_numpy_data("Acrobot-v1")
