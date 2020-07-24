from torch.utils.data import DataLoader
import numpy as np
from collections import deque
import copy
from typing import Union, Dict, Tuple
from minerl.herobraine.hero import spaces
from collections import Counter

from memory.demo_memory import DemoReplayBuffer
from memory.dataset_loader import ExpertDataset
from utils.minerl_wrappers import LazyFramesFlipped, MyLazyFrames
from utils.gen_args import Arguments


def fill_stacked_memory(
        args: Arguments, fill_size, action_space: Union[spaces.Dict, spaces.Discrete],
        pretrain_action_space: Union[spaces.Dict, spaces.Discrete],
        memory: DemoReplayBuffer) -> None:
    """
    Fill memory replay with human observations. Adjust data to match environment input & output.
    :param args: User adjustable constants.
    :param data_dir: Directory of human demonstrations.
    :param fill_size: How many human observations to store in memory.
    :param action_space:
    :param pretrain_action_space:
    :param memory:
    :return: None
    """

    # Use instead of fill_size to get better variance of data
    def collate_fn(batch):
        return batch[0]

    memory.set_pretrain_phase(True)
    env = args.env_name
    if 'Treechop' not in env and 'Dense' not in env:
        env = env.replace('-', 'Dense-')
    data_dir = '/'.join([str(args.minerl_data_root), env])

    # craft_actions = args.crafting_actions
    # camera_actions = args.camera_actions
    # move_actions = args.movement_actions
    expert_dataset = ExpertDataset(data_dir=data_dir, frame_skip=args.frame_skip,
                                   alternate_data='MineRLTreechop-v0' if 'Treechop' not in args.env_name else None)
    expert_data = DataLoader(expert_dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)
    # Use instead of fill_size to get better variance of data

    count = {'step': 0}  # Use dict for mutability
    for states, actions, rewards, next_states, dones, policies, meta, n_policies in expert_data:
        # Add original trajectory
        trajectory = states, actions, rewards, next_states, dones, policies, meta, n_policies
        _add_traj(args, memory, trajectory, fill_size, action_space,
                  pretrain_action_space, count, augment_data=True)

        if count["step"] > fill_size or memory.is_full:
            break

    del expert_data
    print(f'Filled memory with {fill_size} expert observations.')
    print(f"Memory contains {len(memory)} observations.")


def _add_traj(
        args: Arguments, memory: DemoReplayBuffer, trajectory: Tuple, fill_size, action_space,
        pretrain_action_space, count: Dict, augment_data=False):
    states, actions, rewards, next_states, dones, policies, meta, n_policies = trajectory
    frame_skip = args.frame_skip
    tot_frame_period = args.frame_skip * args.frame_stack
    frame_hist = deque([], maxlen=tot_frame_period)
    n_frame_hist = deque([], maxlen=tot_frame_period)
    act_hist = deque([], maxlen=frame_skip)
    pretrain_empty_action = pretrain_action_space.no_op()
    empty_action = action_space.no_op()
    valid_keys = pretrain_empty_action.keys()
    seq_size, d1, d2, ch = states['pov'].shape
    obs_pov_shp = [ch*args.frame_stack, d1, d2]

    # Move actions along since player game has camera acceleration & deceleration.
    # actions['camera'] = np.roll(actions['camera'], shift=-4, axis=0)
    # actions['camera'][-args.frame_skip:] = actions['camera'][-args.frame_skip - 1]

    actions['camera'] = adj_exp_cam(actions['camera'], frame_skip)
    actions['camera'] = np.around(actions['camera'], decimals=4)
    actions['camera'] = discretise_camera(camera=actions['camera'], bins=args.bins)
    # print(actions['camera'])
    seq_size = len(actions['attack'])
    # if 'Treechop' not in args.env:
    #     ActionConverter.branch(actions)  # Group actions according to adjusted action space
    fixed_actions = [copy.deepcopy(pretrain_empty_action) for _ in range(seq_size)]
    for key, values in actions.items():
        for i in range(seq_size):
            if key == 'camera':
                cam_pitch = values[i, 0]
                cam_yaw = values[i, 1]
                # cam_pitch = find_nearest_idx(args.bins, cam_pitch)
                # cam_yaw = find_nearest_idx(args.bins, cam_yaw)
                fixed_actions[i]['camera_0'] = int(cam_pitch)
                fixed_actions[i]['camera_1'] = int(cam_yaw)
            else:
                if key in valid_keys:
                    fixed_actions[i][key] = values[i]
    actions = fixed_actions

    invs = np.zeros((seq_size, 19), dtype=np.uint8)
    next_invs = np.zeros((seq_size, 19), dtype=np.uint8)
    if 'inventory' in states.keys():
        invs = append_mainhand(states)
        next_invs = append_mainhand(next_states)
        next_invs = np.roll(next_invs, shift=-frame_skip, axis=0)
        next_invs[-frame_skip:] = next_invs[-frame_skip - 1]

    states['pov'] = states['pov'].transpose((0, 3, 1, 2))
    next_states['pov'] = next_states['pov'].transpose((0, 3, 1, 2))

    next_states['pov'] = np.roll(next_states['pov'], shift=-frame_skip, axis=0)
    next_states['pov'][-frame_skip:] = next_states['pov'][-frame_skip - 1]

    # Shift actions to use actions from 4 steps in the future. Need this to see what the player did to get from
    # frame 't' -> 't+4'.
    actions = np.roll(actions, shift=-frame_skip, axis=0)
    actions[-frame_skip:] = actions[-frame_skip - 1]

    rewards = np.roll(rewards, shift=-frame_skip, axis=0)
    rewards[-frame_skip:] = rewards[-frame_skip - 1]

    # From T-4 we are basically done. Every step is placed in seperate N-step tracker.
    dones[-frame_skip:] = dones[-1]
    reward_sums = adj_rew(rewards, 4)

    # Real index value since enumerate will iterate over skipped indexes.
    idx = 0
    # Append episode data
    for state, action, reward, next_state, done, inv, n_inv, policy in zip(
            states['pov'], actions, reward_sums, next_states['pov'], dones, invs, next_invs, policies):
        # Skip no-op states... Might mess with history
        if np.all(state == next_state) and action == pretrain_empty_action and reward == 0 and not done:
            continue
        frame_hist.append(state)
        n_frame_hist.append(next_state)
        act_hist.append(action)
        if len(frame_hist) < tot_frame_period:
            continue
        current_action = copy.deepcopy(act_hist[0])

        store_action = _mod_action(args, act_hist, copy.copy(empty_action), current_action, action_space)

        count["step"] += 1

        frames_list = list(frame_hist)[frame_skip - 1::frame_skip]
        stacked_state = MyLazyFrames(frames_list, obs_pov_shp)

        # n_stacked_state = list(n_frame_hist)[frame_skip - 1::frame_skip]
        # n_stacked_state = LazyFrames(n_stacked_state)

        if not args.is_treechop:
            stacked_state = (stacked_state, inv)
            # n_stacked_state = (n_stacked_state, n_inv)

        append_to_memory(
            memory=memory,
            state=stacked_state,
            action=store_action,
            reward=reward,
            done=done,
            expert=True,
            p_idx=policy,
            skip_step=idx % frame_skip
        )

        if augment_data:
            store_action = flip_yaw(store_action, pretrain_action_space)
            stacked_state = LazyFramesFlipped(frames_list, obs_pov_shp)
            if not args.is_treechop:
                stacked_state = (stacked_state, inv)
            append_to_memory(
                memory=memory,
                state=stacked_state,
                action=store_action,
                reward=reward,
                done=done,
                expert=True,
                p_idx=policy,
                skip_step=4 + idx % frame_skip
            )

        idx += 1

        if count["step"] > fill_size or memory.is_full:
            # for k in range(frame_skip):
            #     memory.stop_current_episode(env_id=k)
            print(f'Filled memory with {fill_size} expert observations.')
            print(f"Memory contains {len(memory)} observations.")
            memory.clear_transition_buffers()
            break
    if count["step"] > fill_size or memory.is_full:
        return None


def append_to_memory(memory: DemoReplayBuffer, state, action, reward, done, expert, p_idx, skip_step):
    memory.append(
        state=state,
        action=action,
        reward=reward,
        done=done,
        expert=expert,
        p_idx=p_idx,
        skip_step=skip_step)
    if done:
        memory.stop_current_episode(env_id=skip_step, p_idx=p_idx)


def _mod_action(args: Arguments, act_hist, store_action, current_action, action_space: spaces.Dict):
    store_action.update({
        key: current_action.pop(key, None) for key in args.camera_actions
    })

    craft_dict = {  # Remove craft actions since they are calculated differently.
        key: current_action.pop(key, None) for key in act_hist[0] if key in args.crafting_actions
    }
    for act in list(act_hist)[1:]:
        for key, value in act.items():
            if key in args.movement_actions:
                current_action[key] += value
            if key in args.crafting_actions and craft_dict[key] < value:  # Check to select best action.
                craft_dict[key] = value
    # Find top actions and update new dict

    if 'movement' not in action_space.spaces:
        top_acts = dict(Counter(current_action).most_common(4))
        store_action.update({key: int(val > 0) for key, val in top_acts.items()})
    else:
        top_acts = dict(Counter(current_action).most_common(2))
        movement_action = action_space.no_op()
        test = copy.copy(top_acts)
        store_action = _single_movement_action(movement_action, top_acts,
                                               store_action['camera_0'], store_action['camera_1'])
        assert store_action is not None, f"Got None for store action from action: {test}"
    if not args.is_treechop:
        for key in reversed(list(args.crafting_actions.keys())):
            if craft_dict[key] > 0:
                val = craft_dict[key]
                if key is "equip":
                    val -= 1
                store_action[key] = val
                break
    return store_action


def _single_movement_action(action, top_acts, pitch, yaw):
    # Camera
    if pitch == 2:
        action['movement'] = 1
        return action
    if pitch == 0:
        action['movement'] = 2
        return action
    if yaw == 2:
        action['movement'] = 3
        return action
    if yaw == 0:
        action['movement'] = 4
        return action

    # Buttons
    key, val = list(top_acts.keys())[0], list(top_acts.values())[0]
    if val == 0:
        return action
    top_acts_keys = list(top_acts.keys())
    f_idx = -1 if 'forward' not in top_acts_keys else top_acts_keys.index('forward')
    if f_idx >= 0:  # Forward action present
        forward = top_acts.pop('forward', None)
        if forward is not None and forward > 0:
            key, val = list(top_acts.keys())[0], list(top_acts.values())[0]
            valid_val = val > 0 and forward-1 <= val <= forward-1
            if key == 'jump' and valid_val:
                action['movement'] = 6
                return action
            elif key == 'sprint' and valid_val:
                action['movement'] = 10
                return action
            elif valid_val:
                action['movement'] = 5
                return action
        else:
            key, val = list(top_acts.keys())[0], list(top_acts.values())[0]
    if key == 'left':
        action['movement'] = 7
        return action
    elif key == 'right':
        action['movement'] = 8
        return action
    elif key == 'back':
        action['movement'] = 9
        return action
    elif key == 'jump':
        action['movement'] = 11
        return action
    else:
        return action


def adj_exp_cam(camera: np.ndarray, frame_stack):
    """
    Take camera actions and average 4 _frames.
    :param camera: Camera observation with pitch and yaw for full episode
    :param frame_stack: number of _frames to stack
    :return:
    """
    new_camera = np.zeros_like(camera)
    # camera = np.vstack((camera, np.zeros_like(camera)[:frame_stack-1]))
    for i in range(camera.shape[0]):
        max_index = min(i + frame_stack, camera.shape[0])
        new_camera[i] = np.mean(camera[i: max_index], 0)
    return new_camera


def adj_rew(rewards: np.ndarray, frame_stack):
    new_rewards = np.zeros_like(rewards)
    new_rewards[0] = rewards[0]
    for idx in range(1, rewards.shape[0]):
        min_index = max(idx - frame_stack + 1, 0)
        new_rewards[idx] = np.sum(rewards[min_index: idx+1])
    return new_rewards


def reward_transform(x):
    # return np.sign(x) * np.log2(1 + np.abs(x))
    return np.clip(x, a_min=0.0, a_max=1.0).astype(np.uint8)
    # return x


def discretise_camera(camera, bins):
    disc_cam_data = np.zeros(camera.shape, dtype=np.uint8)
    disc_cam_data[camera == 0] = np.where(bins == 0)[0][0]
    disc_cam_data[camera > bins[-1]] = len(bins) - 1
    for i in range(1, len(bins) // 2):
        disc_cam_data[(bins[i] <= camera) & (camera < bins[i + 1])] = i
    for i in range(len(bins) // 2, len(bins) - 1):
        disc_cam_data[(bins[i] < camera) & (camera <= bins[i + 1])] = i + 1
    return disc_cam_data


def append_mainhand(states):
    x = np.stack(list(states['inventory'].values()))
    x = np.concatenate(([states['equipped_items']['mainhand']['type']], x)).transpose()
    return np.uint8(x.clip(0, 255))


def flip_yaw(action, action_space):
    half_disc_len = action_space.spaces['camera_1'].n // 2
    action = copy.deepcopy(action)
    cam_yaw = action["camera_1"]
    cam_yaw = -(cam_yaw - half_disc_len) + half_disc_len
    action["camera_1"] = cam_yaw
    return action


if __name__ == '__main__':
    rew_test = np.arange(20) + 1
    print(rew_test)
    print(adj_rew(rew_test, 4))
