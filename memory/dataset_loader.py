import minerl
from minerl.herobraine.hero import spaces
import json
from torch.utils.data import Dataset
import sys
import os
import numpy as np
import cv2
import gym
from copy import copy
from collections import OrderedDict, namedtuple, deque
from dataclasses import dataclass
from pprint import pprint
import itertools


FIX_IRON_CRAFT = True
METADATA_FILE = 'metadata.json'


@dataclass
class CraftingRollout:
    action_type: str
    crafted_amount: int = 0
    crafting_amount: int = 0
    req_crafts: int = 0
    score_to_remove: int = 0

    def reset_counts(self):
        self.crafted_amount = 0
        self.crafting_amount = 0
        self.req_crafts = 0
        self.score_to_remove = 0


class ExpertDataset(Dataset):
    def __init__(self, data_dir, skip_failures=True, frame_skip=4, alternate_data: str = None):
        self.frame_skip = frame_skip
        if 'Dense' in data_dir:
            data_dir = data_dir.replace('Dense', '')
        self.env = data_dir.split('/')[-1]

        reward_threshold = {
            'MineRLTreechop-v0': 64,
            'MineRLNavigate-v0': 100,
            'MineRLNavigateExtreme-v0': 100,
            'MineRLObtainIronPickaxe-v0': 256 + 128 + 64 + 32 + 32 + 16 + 8 + 4 + 4 + 2 + 1,
            'MineRLObtainDiamond-v0': 1024 + 256 + 128 + 64 + 32 + 32 + 16 + 8 + 4 + 4 + 2 + 1,
        }
        self.reward_threshold = reward_threshold[self.env]

        self.folders = []
        scores = []
        total_steps = 0
        for root, dirs, files in os.walk(data_dir):
            if METADATA_FILE in files:
                filename = '/'.join([root, METADATA_FILE])
                try:
                    with open(filename) as r:
                        metadata = json.load(r)
                except Exception as e:
                    print(f"Exception {e} loading: {filename}")
                    raise e
                if metadata['total_reward'] < self.reward_threshold and skip_failures:
                    continue
                scores.append(metadata['total_reward'] / metadata['duration_steps'])
                total_steps += metadata['duration_steps']
                self.folders.append(root)

        alt_data = []
        if alternate_data is not None:
            bonus_data = alternate_data
            bonus_data_rew_threshold = reward_threshold[alternate_data]
            other_data_dir = '/'.join(data_dir.split('/')[:-1]+[bonus_data])
            for root, dirs, files in os.walk(other_data_dir):
                if METADATA_FILE in files:
                    filename = '/'.join([root, METADATA_FILE])
                    with open(filename) as r:
                        metadata = json.load(r)
                    if metadata['total_reward'] < bonus_data_rew_threshold and skip_failures:
                        continue
                    scores.append((metadata['total_reward']) / metadata['duration_steps'])
                    total_steps += metadata['duration_steps']
                    alt_data.append(root)
        self.folders = sorted(self.folders, reverse=True)
        alt_data = sorted(alt_data, reverse=True)
        iters = [iter(self.folders), iter(alt_data)]
        self.folders = list(itertools.chain(map(next, itertools.cycle(iters)), *iters))
        self.dense_folders = None
        if 'Treechop' not in self.env:
            self.dense_folders = [folder.replace('-v0', 'Dense-v0') if 'Obtain' in folder else folder
                                  for folder in self.folders]
        print(f'Found a total of {total_steps} steps according to metadata.')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):  # Mostly from minerl DataPipeline
        #         print()
        current_env = self.folders[idx].split('/')[-2]
        # gym_spec = gym.envs.registration.spec(self.env)
        gym_spec = gym.envs.registration.spec(current_env)
        observation_seq, action_seq, reward_seq, next_observation_seq, \
            done_seq, meta, non_dense_rew = self._get_sequence(idx)
        observation_dict = ExpertDataset.map_to_dict(observation_seq, gym_spec._kwargs['observation_space'])
        action_dict = ExpertDataset.map_to_dict(action_seq, gym_spec._kwargs['action_space'])
        next_observation_dict = ExpertDataset.map_to_dict(next_observation_seq, gym_spec._kwargs['observation_space'])

        rewards = reward_seq[0]
        non_dense_rewards = non_dense_rew[0]
        policy_labels = [-1]*len(rewards)
        # print(meta['stream_name'])
        if 'inventory' in observation_dict:
            ExpertDataset._fix_inventory(
                actions=action_dict, invs=observation_dict['inventory'],
                n_invs=next_observation_dict['inventory'], rewards=rewards
            )

            for i, rew in enumerate(non_dense_rewards):
                if rew:
                    policy_labels[i] = ExpertDataset._policy_change(
                        i, observation_dict['inventory'], next_observation_dict['inventory']
                    )
            policy_labels = list(reversed(policy_labels))
            last_policy = policy_labels[0]
            for i in range(len(policy_labels)):
                policy_update = policy_labels[i]
                policy_labels[i] = last_policy
                if policy_update > -1:
                    last_policy = policy_update
            policy_labels = deque(list(reversed(policy_labels)), maxlen=len(policy_labels))
            policy_updates = policy_labels.copy()
            policy_labels.appendleft(0)
        else:
            policy_labels = np.zeros(len(rewards), dtype=np.uint8)
            policy_updates = np.zeros(len(rewards), dtype=np.uint8)

        policy_labels = np.array(policy_labels)
        # print(policy_labels)
        return (observation_dict, action_dict, rewards, next_observation_dict,
                done_seq[0], policy_labels, meta, policy_updates)

    def _get_sequence(self, idx):
        folder = self.folders[idx]
        dense_state = None
        if self.dense_folders is not None:
            dense_folder = self.dense_folders[idx]
            dense_numpy_path = str(os.path.join(dense_folder, 'rendered.npz'))
            dense_state = np.load(dense_numpy_path, allow_pickle=True)

        video_path = str(os.path.join(folder, 'recording.mp4'))
        numpy_path = str(os.path.join(folder, 'rendered.npz'))
        meta_path = str(os.path.join(folder, METADATA_FILE))

        with open(meta_path) as file:
            meta = json.load(file)

        state = np.load(numpy_path, allow_pickle=True)

        action_dict = OrderedDict([(key, state[key]) for key in state if key.startswith('action_')])
        reward_vec = state['reward'] if dense_state is None else dense_state['reward']
        info_dict = OrderedDict([(key, state[key]) for key in state if key.startswith('observation_')])

        non_dense_rew = state['reward']

        num_states = len(reward_vec) + 1

        try:
            # Start video decompression
            cap = cv2.VideoCapture(video_path)

            ret, frame_num = True, 0
            while ret:
                ret, _ = ExpertDataset.read_frame(cap)
                if ret:
                    frame_num += 1
            cap.release()

            max_frame_num = frame_num  # int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <- this is not correct!
            frames = []
            frame_num, stop_idx = 0, 0

            cap = cv2.VideoCapture(video_path)
            for _ in range(max_frame_num - num_states):
                ret, _ = ExpertDataset.read_frame(cap)
                frame_num += 1
                if not ret:
                    return None

            # Rendered Frames
            observables = list(info_dict.keys()).copy()
            observables.append('pov')
            actionables = list(action_dict.keys())

            # Loop through the video and construct _frames
            # of observations to be sent via the multiprocessing queue
            # in chunks of worker_batch_size to the batch_iter loop.
            ret = True
            start_idx = stop_idx

            # Collect up to worker_batch_size number of _frames
            try:
                # Go until max_seq_len +1 for S_t, A_t,  -> R_t, S_{t+1}, D_{t+1}
                while ret and frame_num < max_frame_num:
                    ret, frame = ExpertDataset.read_frame(cap)
                    frames.append(frame)
                    frame_num += 1
                cap.release()

            except Exception as err:
                print("error reading capture device:", err)
                raise err

            if frame_num == max_frame_num:
                frames[-1] = frames[-2]

            # Next sarsd pair index
            stop_idx = start_idx + len(frames) - 1
            # print('Num _frames in batch:', stop_idx - start_idx)

            # Load non-image data from npz
            current_observation_data = [None for _ in observables]
            action_data = [None for _ in actionables]
            next_observation_data = [None for _ in observables]

            try:
                for i, key in enumerate(observables):
                    if key == 'pov':
                        current_observation_data[i] = np.asanyarray(frames[:stop_idx])
                        next_observation_data[i] = np.asanyarray(frames[1:stop_idx + 1])
                    elif key == 'observation_compassAngle':
                        current_observation_data[i] = np.asanyarray(info_dict[key][start_idx:stop_idx, 0])
                        next_observation_data[i] = np.asanyarray(info_dict[key][start_idx + 1:stop_idx + 1, 0])
                    else:
                        current_observation_data[i] = np.asanyarray(info_dict[key][start_idx:stop_idx])
                        next_observation_data[i] = np.asanyarray(info_dict[key][start_idx + 1:stop_idx + 1])

                # We are getting (S_t, A_t -> R_t),   S_{t+1}, D_{t+1} so there are less actions and rewards
                for i, key in enumerate(actionables):
                    action_data[i] = np.asanyarray(action_dict[key][start_idx: stop_idx])

                reward_data = np.asanyarray(reward_vec[start_idx:stop_idx], dtype=np.float32)
                non_dense_rew = np.asanyarray(non_dense_rew[start_idx:stop_idx], dtype=np.float32)
                # reward_data = np.clip(reward_data, a_min=0, a_max=1)

                done_data = [False for _ in range(len(reward_data))]
                if frame_num == max_frame_num:
                    done_data[-1] = True
            except Exception as err:
                print("error drawing batch from npz file:", err)
                raise err

            batches = [current_observation_data, action_data, [reward_data], next_observation_data,
                       [np.array(done_data, dtype=np.bool)], meta, [non_dense_rew]]
        except BrokenPipeError:

            print("Broken pipe!")
            return None
        except FileNotFoundError as e:
            print("File not found!")
            raise e
        except Exception as e:
            print("Exception \'{}\' caught on file \"{}\" by a worker of the data pipeline.".format(e, folder))
            return None
        return batches

    @staticmethod
    def _fix_inventory(actions, invs, n_invs, rewards, frame_skip=4):
        """
        Very sloppy code... However it is necessary to deal with flaws in demo data.
        :param actions:
        :param invs:
        :param n_invs:
        :param rewards:
        :param frame_skip:
        :return:
        """
        furnace_placed = False
        logs_arr = invs['log'].copy()
        n_logs_arr = n_invs['log'].copy()
        planks_arr = invs['planks'].copy()
        n_planks_arr = n_invs['planks'].copy()

        ore_arr = invs['iron_ore'].copy()
        n_ore_arr = n_invs['iron_ore'].copy()
        ingot_arr = invs['iron_ingot'].copy()
        n_ingot_arr = n_invs['iron_ingot'].copy()

        plank_rollout = CraftingRollout(action_type='craft')
        plank_start_idx = -1

        iron_rollout = CraftingRollout(action_type='nearbySmelt')
        iron_start_idx = -1
        episode_ingot_diff = ingot_arr - n_ingot_arr  # Find used ingots

        total_ingots = np.sum(episode_ingot_diff[episode_ingot_diff > 0])
        ore_diff = (n_ore_arr - ore_arr).clip(0)
        total_ores = ore_diff.sum() - total_ingots

        for idx, reward in enumerate(rewards):
            if reward > 0 and actions['craft'][idx] == 3:
                # Player crafted multiple planks in one step. Need to roll out.
                if plank_rollout.req_crafts == 0:  # Check for in progress rollouts.
                    plank_rollout.reset_counts()
                    plank_rollout.crafting_amount = invs['log'][idx-1]
                    plank_rollout.crafted_amount = invs['planks'][idx-1]
                    plank_start_idx = idx
                    plank_rollout.req_crafts = int(reward // 8)
                else:
                    plank_rollout.req_crafts += int(reward // 8)

            if plank_rollout.req_crafts > 0:
                fix_plank_craft(idx, plank_rollout, actions, logs_arr, n_logs_arr, planks_arr, n_planks_arr, rewards,
                                do_craft=(idx-plank_start_idx) % frame_skip == 0)

            if not FIX_IRON_CRAFT:
                continue

            ###############################################################################

            if invs['iron_ore'][idx] < n_invs['iron_ore'][idx] and \
                    iron_rollout.req_crafts == 0 and iron_rollout.score_to_remove <= 0:
                # Player started crafting iron ore...
                # Check for in progress rollouts.
                # iron_rollout.reset_counts()
                iron_start_idx = idx
                # iron_rollout.crafting_amount = invs['iron_ore'][idx]
                iron_rollout.crafted_amount = invs['iron_ingot'][idx]
                # iron_rollout.req_crafts = n_invs['iron_ore'][idx]
                iron_rollout.score_to_remove = total_ingots*128 + total_ores*64
                # continue

            if reward > 0 and iron_rollout.score_to_remove > 0 \
                    and reward % 128 == 0 and invs['iron_ingot'][idx] < n_invs['iron_ingot'][idx]:
                rewards[idx] = 0
                iron_rollout.score_to_remove -= int(128*max(0, n_invs['iron_ingot'][idx] - invs['iron_ingot'][idx]))

            if reward > 0 and reward % 64 == 0 and invs['iron_ore'][idx] > invs['iron_ore'][idx-1]:
                # print(f"Step: {idx};  Inv: {invs['iron_ore'][idx]};  Prev next inv: {invs['iron_ore'][idx-1]}")
                if furnace_placed:  # Nowhere to place ores
                    rewards[idx] = 0
                    iron_rollout.score_to_remove -= int(64*max(0, invs['iron_ore'][idx] - invs['iron_ore'][idx-1]))
                else:
                    ore_arr[idx] -= int(rewards[idx] // 64)

            if iron_rollout.score_to_remove > 0:
                fix_iron_smelt(idx, iron_rollout, actions, ore_arr, n_ore_arr, ingot_arr, n_ingot_arr, rewards,
                               do_craft=furnace_placed and (idx-iron_start_idx) % frame_skip == 0)

            if reward > 0 and reward % 64 == 0 and \
                    invs['iron_ore'][idx] < n_invs['iron_ore'][idx]:
                ore_obtained = int(max(0, n_invs['iron_ore'][idx] - invs['iron_ore'][idx]))
                iron_rollout.req_crafts += ore_obtained
                iron_rollout.crafting_amount += ore_obtained
                n_ore_arr[idx] += ore_obtained

            if actions['place'][idx] == 5:
                furnace_placed = True

            if n_invs['furnace'][idx] - invs['furnace'][idx] == 1:
                furnace_placed = False

        invs['log'] = logs_arr
        n_invs['log'] = n_logs_arr
        invs['planks'] = planks_arr
        n_invs['planks'] = n_planks_arr

        invs['iron_ore'] = ore_arr
        n_invs['iron_ore'] = n_ore_arr
        invs['iron_ingot'] = ingot_arr
        n_invs['iron_ingot'] = n_ingot_arr

    @staticmethod
    def read_frame(cap):  # From minerl DataPipeline
        try:
            ret, frame = cap.read()
            if ret:
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)

            return ret, frame
        except Exception as err:
            print("error reading capture device:", err)
            raise err

    @staticmethod
    def map_to_dict(handler_list: list, target_space: gym.spaces.space):

        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
            if isinstance(gym_space, spaces.Dict):
                dont_count = False
                inner_dict = OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):
                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)
                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]
                return i + 1

        result = OrderedDict()
        index = 0
        for key, space in target_space.spaces.items():
            index = _map_to_dict(index, handler_list, key, space, result)
        return result

    @staticmethod
    def _policy_change(idx, inv_arr, n_inv_arr):
        inv = copy(inv_arr)
        n_inv = copy(n_inv_arr)
        for key, value in inv.items():
            inv[key] = value[idx]
        for key, value in n_inv.items():
            n_inv[key] = value[idx]
        policy = detect_policy_change(inv, n_inv) - 1
        # if policy == 8:
        #     print(f"Step: {idx}; {inv['stone_pickaxe']}; {n_inv['stone_pickaxe']}")

        # if policy == 10:
        #     print(idx)

        # if policy == 9 and idx != len(inv_arr['log'])-1:
        #     policy = 0
        return policy


def detect_policy_change(inv, n_inv):
    if inv['log'] < n_inv['log']:
        return 1
    if inv['planks'] < n_inv['planks']:
        return 2
    if inv['stick'] < n_inv['stick'] or inv['crafting_table'] < n_inv['crafting_table']:
        return 3
    if inv['wooden_pickaxe'] < n_inv['wooden_pickaxe']:
        return 4
    if inv['cobblestone'] < n_inv['cobblestone']:
        return 5
    if inv['furnace'] < n_inv['furnace'] or inv['stone_pickaxe'] < n_inv['stone_pickaxe']:
        return 6
    if inv['iron_ore'] < n_inv['iron_ore']:
        return 7
    if inv['iron_ingot'] < n_inv['iron_ingot']:
        return 8
    if inv['iron_pickaxe'] < n_inv['iron_pickaxe']:
        return 9
    # if reward == 1024:
    return 9


def fix_plank_craft(idx, plank_rollout: CraftingRollout, actions, logs_arr, n_logs_arr,
                    planks_arr, n_planks_arr, rewards: np.ndarray, do_craft=True):
    logs_arr[idx] = plank_rollout.crafting_amount
    planks_arr[idx] = plank_rollout.crafted_amount
    if do_craft:
        plank_rollout.crafting_amount -= 1
        plank_rollout.crafted_amount += 4
        plank_rollout.req_crafts -= 1
        actions[plank_rollout.action_type][idx] = 3
        rewards[idx] = 8
    # if plank_rollout.req_crafts == 0:
        # Only need to change next values once... Since they are linked to the previous current value...
    n_logs_arr[idx] = plank_rollout.crafting_amount
    n_planks_arr[idx] = plank_rollout.crafted_amount


def fix_iron_smelt(idx, iron_rollout: CraftingRollout, actions, ore_arr, n_ore_arr,
                   ingot_arr, n_ingot_arr, rewards: np.ndarray, do_craft=True):
    ore_arr[idx] = iron_rollout.crafting_amount
    ingot_arr[idx] = iron_rollout.crafted_amount
    if do_craft and iron_rollout.req_crafts > 0:
        iron_rollout.crafting_amount -= 1
        iron_rollout.crafted_amount += 1
        iron_rollout.req_crafts -= 1
        actions[iron_rollout.action_type][idx] = 1
        rewards[idx] = 128
    # if iron_rollout.score_to_remove <= 0:
        # Only need to change next values once... Since they are linked to the previous current value...
    n_ore_arr[idx] = iron_rollout.crafting_amount
    n_ingot_arr[idx] = iron_rollout.crafted_amount


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_dir = os.getenv('MINERL_DATA_ROOT', 'data/')
    env = 'MineRLObtainIronPickaxe-v0'
    # env = 'MineRLObtainDiamondDense-v0'
    data_dir = '/'.join([str(data_dir), env])

    # Use instead of fill_size to get better variance of data
    def collate_fn(batch):
        return batch[0]

    expert_dataset = ExpertDataset(data_dir=data_dir)
    data = DataLoader(expert_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for idx, *_ in enumerate(data):
        print(f'Tested demo number {idx}.\n')
