import statistics
import copy

from scipy import stats
import gym
import minerl
import numpy as np
from torch.utils.data import DataLoader

from memory.dataset_loader import ExpertDataset
from utils.gen_args import Arguments
from utils.minerl_wrappers import wrap_minerl_obs_space, wrap_minerl_action_space


def get_camera_discretizations(args: Arguments, action_space: minerl.env.spaces.Dict):
    def collate_fn(batch):
        return batch[0]

    empty_action = action_space.no_op()
    valid_keys = list(empty_action.keys())
    valid_keys.append("camera")
    valid_keys.remove("camera_0")
    valid_keys.remove("camera_1")

    env = args.env_name
    if 'Treechop' not in env and 'Dense' not in env:
        env = env.replace('-', 'Dense-')
    data_dir = '/'.join([str(args.minerl_data_root), env])
    expert_dataset = ExpertDataset(data_dir=data_dir, frame_skip=4,
                                   alternate_data='MineRLTreechop-v0' if 'Treechop' not in args.env_name else None)
    data = DataLoader(expert_dataset, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn)
    cam_pitch_arr = []
    cam_yaw_arr = []

    for states, actions, rewards, next_states, dones, policies, meta, n_policies in data:
        actions['camera'] = adj_exp_cam(actions['camera'], 4)
        actions['camera'] = np.around(actions['camera'], decimals=4)
        # actions['camera'] = discretise_camera(camera=actions['camera'], bins=args.bins[0])
        # print(actions['camera'])
        seq_size = len(actions['attack'])
        # if 'Treechop' not in args.env:
        #     ActionConverter.branch(actions)  # Group actions according to adjusted action space
        fixed_actions = [copy.deepcopy(empty_action) for _ in range(seq_size)]
        for key, values in actions.items():
            for i in range(seq_size):
                if key in valid_keys:
                    fixed_actions[i][key] = values[i]

        actions = fixed_actions

        for state, action, reward, next_state, done in zip(
                states['pov'], actions, rewards, next_states['pov'], dones):

            if np.all(state == next_state) and action == empty_action and reward == 0 and not done:
                continue
            cam_pitch_arr.append(action['camera'][0])
            cam_yaw_arr.append(action['camera'][1])

    return cam_pitch_arr, cam_yaw_arr


def get_env_spaces(args: Arguments):
    minerl_env: bool = 'MineRL' in args.env_name
    env_spec = gym.envs.registry.spec(args.env_name)
    observation_space = env_spec._kwargs['observation_space']  # Private variable contains what we need
    action_space = env_spec._kwargs['action_space']  # Private variable contains what we need
    if minerl_env:
        observation_space = wrap_minerl_obs_space(observation_space)
        action_space = wrap_minerl_action_space(args, action_space)
    return observation_space, action_space


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


if __name__ == '__main__':
    args = Arguments(underscores_to_dashes=True).parse_args(known_only=False)
    observation_space, action_space = get_env_spaces(args)
    cam_pitch, cam_yaw = get_camera_discretizations(args, action_space)
    cam_pitch, cam_yaw = np.array(cam_pitch), np.array(cam_yaw)
    cam_pitch, cam_yaw = cam_pitch[cam_pitch != 0], cam_yaw[cam_yaw != 0]
    all_cam = np.append(cam_pitch, cam_yaw)

    print(".1% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, .1)}")
    print(f"Yaw:   {np.percentile(cam_yaw,  .1)}")
    print(f"All:   {np.percentile(all_cam,  .1)}\n")

    print("1% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 1)}")
    print(f"Yaw:   {np.percentile(cam_yaw,  1)}")
    print(f"All:   {np.percentile(all_cam,  1)}\n")

    print("5% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 5)}")
    print(f"Yaw:   {np.percentile(cam_yaw,  5)}")
    print(f"All:   {np.percentile(all_cam,  5)}\n")

    print("25% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 25)}")
    print(f"Yaw:   {np.percentile(cam_yaw,  25)}")
    print(f"All:   {np.percentile(all_cam,  25)}\n")

    print("40% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 40)}")
    print(f"Yaw:   {np.percentile(cam_yaw,  40)}")
    print(f"All:   {np.percentile(all_cam,  40)}\n")

    print("Median:")
    print(f"Pitch: {np.percentile(cam_pitch, 50)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 50)}")
    print(f"All:   {np.percentile(all_cam, 50)}\n")

    print("60% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 60)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 60)}")
    print(f"All:   {np.percentile(all_cam, 60)}\n")

    print("75% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 75)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 75)}")
    print(f"All:   {np.percentile(all_cam, 75)}\n")

    print("95% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 95)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 95)}")
    print(f"All:   {np.percentile(all_cam, 95)}\n")

    print("99% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 99)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 99)}")
    print(f"All:   {np.percentile(all_cam, 99)}\n")

    print("99.9% quartile:")
    print(f"Pitch: {np.percentile(cam_pitch, 99.9)}")
    print(f"Yaw:   {np.percentile(cam_yaw, 99.9)}")
    print(f"All:   {np.percentile(all_cam, 99.9)}\n")

    cam_pitch_std = cam_pitch.std()
    print(cam_pitch_std)
    cam_yaw_std = cam_yaw.std()
    print(cam_yaw_std)
    mean = 0

    print([i*cam_pitch_std for i in range(-3, 4)])
    print([i*cam_yaw_std for i in range(-3, 4)])
