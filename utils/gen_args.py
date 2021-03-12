#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from typing import List, Tuple

try:
    import argcomplete
except ImportError as e:
    print('Install argcomplete to enable command line option autocomplete.')
import os
import sys
import torch
from tap import Tap
import json
import tempfile
from datetime import datetime
from typing_extensions import Literal
import numpy as np
import gym
import minerl

CAM_DISC = np.array([-5.0/4, 0.0, 5.0/4])
REWARD_RANGES = {
    'Acrobot-v1': (-500, 0),
    'CartPole-v0': (0, 200)
}


class Arguments(Tap):
    experiment_name: str = None
    # noinspection PyUnresolvedReferences
    env_name: Literal[
        'PongNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'MineRLTreechop-v0',
        'MineRLObtainIronPickaxeDense-v0',
        'CartPole-v0',
        'Acrobot-v1'
    ] = 'CartPole-v0'
    rew_range: Tuple[float, float] = None

    # Weights & Biases args
    log_run: bool = False  # Log with wandb
    local_log: bool = False  # Log locally in csv files

    # Training arguments
    batch_size: int = 64  # Batch size of offline training
    train_steps: int = 1500000  # Total steps to train for
    test: bool = False  # Test saved agent
    resume: bool = False  # Resume training
    render: bool = False  # Render env
    monitor: bool = False  # Save recordings of episodes.
    outdir: str = 'results'  # Directory path to save output files. If it does not exist, it will be created.
    verbosity: int = 2  # Verbosity levels: 0 = no printing; 1 = only progressbar; 2 = progress + episodic
    train_episodes: int = 1500

    # NN args
    nn_hidden_layers: List[int] = [1024, 512, 256]

    # Default DQN params
    gamma: float = 0.99  # Discount factor of DQN training
    seed: int = 2147483646  # Seed for all RNG packages
    fix_seed: bool = False  # Fix seed for all runs
    replay_frequency: int = 4  # How often to run offline training (every N steps)
    learn_start: int = 50000  # How many steps to observe agent training
    frame_skip: int = 4  # Number of frames to skip per step
    frame_stack: int = 4  # Number of frames to stack
    batch_accumulator: str = 'mean'  # How to accumulate a batch of losses sum or mean

    # Deep learning arguments
    lr: float = 0.0000625  # Learning rate of deep learning
    save_freq: int = 2000  # Save model every N steps
    device = None  # Device to train model on. Defaults to GPU if available

    # Double DQN
    no_double_dqn: bool = False  # Use Double DQN
    target_update: int = 10000  # When to update target network

    # Dueling DQN
    no_dueling: bool = False  # Use dueling DQN

    # N-step DQN
    n_step: int = 10  # N-step DQN, default DQN when N = 1

    # Replay memory
    memory_capacity: int = 3000000  # Capacity of memory

    # Prioritized experience replay
    no_prioritized: bool = False  # Don't use Prioritized Experience Replay

    # e-greedy
    greedy: bool = False  # Act only greedy (Use either this or noisy or both, otherwise no exploration)
    epsilon_steps: int = 100000  # Steps to anneal epsilon to its final value
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01  # Final epsilon value

    # Noisy
    noisy: bool = False  # Use noisy linear layers in networks
    noise_std: float = 0.5  # Standard noise value (not implemented)

    # C51
    no_c51: bool = False  # Activate C51 algo
    atoms: int = 51  # Number of atoms to use, recommended to stick to 51
    v_min: float = -10.0  # Minimum reward for C51 (clips lower values)
    v_max: float = 10.0  # Maximum reward for C51 (clips higher values)

    # Automatically initiated variables
    double_dqn: bool = None
    prioritized: bool = None
    dueling: bool = None
    use_c51: bool = None

    ####################################################################################################################
    #                                               MineRL specific                                                    #
    ####################################################################################################################
    # Action branching
    no_action_branching: bool = False  # Deactivate action branching

    # DQfD
    lambda0: float = 1.0  # Lambda 0 for J1.
    lambda1: float = 1.0  # Lambda 0 for Jn.
    lambda2: float = 1.0  # Lambda 0 for Je.
    lambda3: float = 1e-5  # Lambda 0 for JL2.
    bonus_priority_agent: float = 0.001  # Bonus priority for agent observations.
    bonus_priority_demo: float = 1.0  # Bonus priority for demo observations.
    dqfd_loss: bool = False  # Use DQfD loss (Jdq + Jn + Je + Jl2)
    skip_pretrain: bool = False  # For skipping pre-training
    no_expert_memory: bool = False
    pretrain_steps: int = 80000  # Batches to pre-train for
    margin_loss: float = 0.4  # Margin loss value.
    expert_fraction: float = 0.3  # Fraction of memory capacity dedicated to expert observations.
    pretrain_batch_size: int = 128  # Batch size when pre-training

    # ForgER
    use_forget: bool = False  # Control the decrease in expert data usage per batch
    forget_min: float = 0.5  # Minimum expert data to use in a batch
    forget_final_step: int = 500000  # Final step to settle on min forget %

    # Saliency
    saliency_maps: bool = False
    save_saliency: bool = False

    # Automatically initiated variables
    bins = None
    movement_actions = None
    exclude_actions = None
    camera_actions = None
    crafting_actions = None
    minerl_data_root = None
    is_treechop = None
    action_branching = None

    def process_args(self) -> None:
        assert self.n_step >= 1, f"n-step needs to be bigger or equal to 1. N=1 is default DQN."
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.rew_range = REWARD_RANGES[self.env_name]
        # self._set_min_max()
        print(f"Reward range [{self.v_min}, {self.v_max}]")

        self.double_dqn = not self.no_double_dqn
        self.prioritized = not self.no_prioritized
        self.dueling = not self.no_dueling
        self.action_branching = not self.no_action_branching
        self.use_c51 = not self.no_c51

        if self.env_name in ['CartPole-v0', 'Acrobot-v1']:  # Simple envs
            self.frame_skip = 1
            self.frame_stack = 1
            self.nn_hidden_layers = [64, 32, 16]
            self.pretrain_steps = 15000
            self.forget_min = 0.0
            self.learn_start = 500
            self.forget_final_step = 10000
            self.epsilon_final = 0.01
            self.epsilon_start = 0.75
            self.epsilon_steps = 4000
            self.n_step = 6
        if self.no_expert_memory:
            self.learn_start = 5000
            self.epsilon_start = 1.0
            self.epsilon_steps = 100000
            self.lambda3 = 0
            self.expert_fraction = 0

        if self.saliency_maps:
            self.test = True
            self.outdir = prepare_output_dir(self.outdir)

        if 'MineRL' in self.env_name:
            self.v_min: float = 0.0
            self.v_max: float = 50.0
            self.lr = 0.0000625
            self.train_steps = 8000000 // 4
            self.learn_start //= 10
            if self.skip_pretrain:
                self.pretrain_steps = 0
            self.minerl_data_root = os.getenv('MINERL_DATA_ROOT', 'data/')
            self.movement_actions = ["attack", "back", "forward", "jump", "left", "right", "sneak", "sprint"]
            # self.exclude_actions = ['back', 'left', 'right', 'sneak']
            self.exclude_actions = ['sneak']
            self.movement_actions = [act for act in self.movement_actions if act not in self.exclude_actions]
            self.camera_actions = ["camera_0", "camera_1"]
            self.crafting_actions = ['place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']
            self.is_treechop = True

            if self.action_branching:
                self.bins = np.array([-10.0, -4.0, -2.0, -0.125, 0.0, 0.125, 2.0, 4.0, 10.0])
            else:
                self.bins = CAM_DISC
            if self.monitor:
                self.env_name = use_custom_env(self.env_name)
                self.outdir = prepare_output_dir(self.outdir)

        if 'argcomplete' in sys.modules:
            argcomplete.autocomplete(self)

    def _set_min_max(self):
        possible_val_range = self.atoms - 11
        min_rew, max_rew = self.rew_range
        rew_val_range = abs(max_rew - min_rew)
        val_range = possible_val_range if rew_val_range > possible_val_range else rew_val_range
        if max_rew <= 0:
            self.v_max = max_rew
            self.v_min = max_rew - val_range
        elif min_rew >= 0:
            self.v_min = min_rew
            self.v_max = min_rew + val_range
        elif min_rew < 0 < max_rew:
            self.v_min = -val_range // 2
            self.v_max = val_range // 2


def prepare_output_dir(user_specified_dir, argv=None,
                       time_format='%d-%m-%y %H:%M:%S'):
    time_str = datetime.now().strftime(time_format)
    print('Making dir!')
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir) and not os.path.isdir(user_specified_dir):
            raise RuntimeError('{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.txt'), 'w') as f:
        f.write(json.dumps(dict(os.environ)))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        if argv is None:
            argv = sys.argv
        f.write(' '.join(argv))

    return outdir


def use_custom_env(original_env_id: str, wh_size: int = 512) -> str:
    from gym.envs.registration import register

    xml_name = 'treechop'
    env_spec = gym.envs.registry.spec(original_env_id)
    observation_space = env_spec._kwargs['observation_space']  # Private variable contains what we need
    observation_space.spaces['pov'].shape = (wh_size, wh_size, 3)
    action_space = env_spec._kwargs['action_space']  # Private variable contains what we need
    max_steps = env_spec.max_episode_steps

    env_spec = env_spec._kwargs['env_spec']
    env_spec.resolution = (wh_size, wh_size)
    env_spec._observation_space = observation_space

    if 'iron' in original_env_id.lower():
        xml_name = 'obtainIronPickaxe'
    elif 'diamond' in original_env_id.lower():
        xml_name = 'obtainDiamond'
    if 'dense' in original_env_id.lower():
        xml_name += 'Dense'

    custom_env_id = original_env_id.replace('v0', 'v99')
    cwd = os.getcwd()
    xml_dir = 'recording_xml'
    my_mission_dir = '/'.join([cwd, 'utils', xml_dir])
    register(
        id=custom_env_id,
        entry_point='minerl.env:MineRLEnv',
        kwargs={
            'xml': os.path.join(my_mission_dir, ''.join([xml_name, '.xml'])),
            'observation_space': observation_space,
            'action_space': action_space,
            'env_spec': env_spec
        },
        max_episode_steps=max_steps,
    )
    return custom_env_id


if __name__ == '__main__':
    print(use_custom_env("MineRLObtainDiamond-v0"))
