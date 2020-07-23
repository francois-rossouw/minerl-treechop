#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from typing import Union, Type
import gym
import minerl
import wandb
import torch

from agents import DQN, C51, fill_stacked_memory
from utils import (Arguments, Logger, seed_things, make_env, make_minerl_env,
                   wrap_minerl_obs_space, wrap_minerl_action_space)
from memory import PrioritizedExperienceBuffer, UniformExperienceBuffer, DemoReplayBuffer


def main_train(args: Arguments):
    minerl_env: bool = 'MineRL' in args.env_name
    seed_things(args.seed)
    logger = Logger(args.log_run)
    if args.log_run:
        generate_wandb(args)
    if args.local_log:
        logger.create_datawriter(args)

    # Get action space of env
    observation_space, action_space, pretrain_action_space = get_env_spaces(args)
    print(action_space)
    print(observation_space)

    # Create agent
    args.device = torch.device(args.device)
    agent_cls: Type[DQN] = get_agent_cls(args)
    agent = agent_cls(
        args=args, logger=logger, action_space=action_space,
        observation_space=observation_space
    )
    print(agent.online_net)

    logger.init_tracker()
    if args.test or args.greedy:
        logger.epsilon = args.epsilon_final

    memory = None
    if not args.test:
        mem_cls = get_memory_cls(args)
        betasteps = (args.pretrain_steps + args.train_steps // 10) / args.replay_frequency
        expert_capacity = int(args.memory_capacity * args.expert_fraction)
        memory_capacity = args.memory_capacity - expert_capacity
        memory = mem_cls(
            capacity=memory_capacity,
            demo_capacity=expert_capacity,
            n_step=args.n_step,
            betasteps=betasteps,
            gamma=args.gamma,
            ir_prob=args.ir_prob,
            prioritized=args.prioritized,
            bonus_priority_demo=args.bonus_priority_demo,
            bonus_priority_agent=args.bonus_priority_agent
        )

        fill_stacked_memory(
            args=args, fill_size=args.memory_capacity, action_space=action_space,
            pretrain_action_space=pretrain_action_space, memory=memory)

        if not args.skip_pretrain:
            agent.pretrain(args, action_space, pretrain_action_space, memory)

    env = make_minerl_env(args.env_name, args) if minerl_env else make_env(args.env_name)
    try:
        if not args.test:
            agent.train_agent(args, env, memory)
        else:
            agent.test_agent(args, env)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def get_env_spaces(args: Arguments):
    minerl_env: bool = 'MineRL' in args.env_name
    env_spec = gym.envs.registry.spec(args.env_name)
    observation_space = env_spec._kwargs['observation_space']  # Private variable contains what we need
    action_space = env_spec._kwargs['action_space']  # Private variable contains what we need
    pretrain_action_space = None
    if minerl_env:
        observation_space = wrap_minerl_obs_space(observation_space)
        action_space, pretrain_action_space = wrap_minerl_action_space(args, action_space)
    return observation_space, action_space, pretrain_action_space


def get_memory_cls(args: Arguments) -> Type[DemoReplayBuffer]:
    # Type[Union[PrioritizedExperienceBuffer, UniformExperienceBuffer, DemoReplayBuffer]]:
    return DemoReplayBuffer
    # if not args.skip_pretrain:
    # if args.prioritized:
    #     return PrioritizedExperienceBuffer
    # else:
    #     return UniformExperienceBuffer


def get_agent_cls(args: Arguments) -> Type[DQN]:
    if args.use_c51:
        return C51
    else:
        return DQN


def generate_wandb(args: Arguments):
    tags = []
    if args.double_dqn:
        tags.append("double-dqn")
    if args.dueling:
        tags.append("dueling-dqn")
    if args.n_step > 1:
        tags.append(f"{args.n_step}-step")
    if not args.greedy:
        tags.append("e-greedy")
    if args.noisy:
        tags.append("noisy")
    if args.prioritized:
        tags.append("prioritized-experience-replay")

    wandb.init(
        project="Rainbow DQN",
        config=args.as_dict(),
        tags=tags,
        job_type="training" if not args.test else "eval",
        monitor_gym=args.monitor,
        group=args.env_name
    )


def set_def_rainbow(args: Arguments):
    args.use_c51 = True
    # args.noisy = True
    args.greedy = True
    # args.verbosity = 0
    # args.log_run = True
    args.n_step = 10
    # args.lambda0 = 0.5
    # args.lambda1 = 0.5
    args.dqfd_loss = True
    # args.fix_seed = True


def _project_profile_settings(args: Arguments):
    args.pretrain_steps = 8000
    args.train_steps = 20000
    args.verbosity = 1


if __name__ == '__main__':
    # args = Arguments(underscores_to_dashes=True).parse_args(known_only=False)
    # main_train(args)

    ####################################################################################################################
    #                                           FOR RUNNING SWEEP                                                      #
    ####################################################################################################################
    args = Arguments(underscores_to_dashes=True).parse_args(known_only=False)
    set_def_rainbow(args)
    # _project_profile_settings(args)
    main_train(args)
