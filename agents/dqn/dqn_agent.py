import os
from typing import Tuple, Union, List, Type
from collections import OrderedDict
import itertools
from statistics import mean

from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch import optim
import numpy as np
import gym
from gym import spaces

from agents.dqn.models import LinearDQNModel, ConvDQNModel
from agents.common import bc_loss, AgentAbstract
from memory.prioritized_memory import PrioritizedExperienceBuffer
from memory.demo_memory import DemoReplayBuffer
from memory.common import ExperienceSamples
from utils.gen_args import Arguments
from utils.utilities import totensor, soft_update, unsqueeze_obs, print_b, generate_saliency_map
from utils.checkpointing import save_checkpoint, load_model
from utils.mylogger import Logger
from utils.minerl_wrappers import TupleSpace


class DQN(AgentAbstract):
    def __init__(self, args: Arguments, logger: Logger, action_space, observation_space: gym.spaces.Box, p_idx=0, **kwargs):
        """
        Main DQN agent class
        :param args: all default arguments. Can be changed in cmd line
        :param logger: logging object
        :param action_space: action space of gym env
        :param observation_space: observation space of gym env
        :param p_idx: policy index
        :return: self
        """
        self.p_idx = p_idx
        self.arch = "DQN"
        if args.double_dqn:
            self.arch = f"Double{self.arch}"
        if args.dueling:
            self.arch = f"Dueling{self.arch}"

        self.action_space = action_space
        self.is_branched = args.action_branching
        self.observation_space = observation_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.cam_branch_idxs = None
            self.n_actions = [action_space.n]
            self.nr_action_branches = 1
        elif not self.is_branched:
            self.cam_branch_idxs = None
            self.n_actions = [action_space.spaces['movement'].n]
            self.nr_action_branches = 1
        else:
            action_space_keys = list(action_space.spaces.keys())
            self.cam_branch_idxs = [
                action_space_keys.index('camera_0'), action_space_keys.index('camera_1')
            ]
            self.n_actions = list(
                action.n for action in action_space.spaces.values()
            )
            self.nr_action_branches = len(self.n_actions)
        self.logger = logger
        self._set_args(args)
        self._model_type = LinearDQNModel if len(observation_space.shape) == 1 else ConvDQNModel

        if type(self) == DQN:
            # Only create models if not called from a subclass
            self._create_models(args, self.n_actions, observation_space, self._model_type)

    def _set_args(self, args: Arguments):
        self.batch_size = args.batch_size
        self.greedy = args.greedy
        self.device = args.device
        self.n_step = args.n_step
        self.gamma = args.gamma
        self.test = args.test
        self.resume = args.resume
        self.save_freq = args.save_freq
        self.learn_start = args.learn_start
        self.replay_frequency = args.replay_frequency
        self.epsilon_start = args.epsilon_start
        self.epsilon_final = args.epsilon_final
        self.epsilon_steps = args.epsilon_steps
        self.dueling = args.dueling
        self.double_dqn = args.double_dqn
        self.target_update = args.target_update
        self.noisy = args.noisy
        self.prioritized = args.prioritized
        self.lambda0 = args.lambda0
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.dqfd_loss = args.dqfd_loss
        self.use_forget = args.use_forget
        self.forget_final_step = args.forget_final_step
        self.forget_min = args.forget_min
        self.batch_accumulator = args.batch_accumulator
        self.margin_loss = args.margin_loss
        self.gen_saliency_maps = args.saliency_maps
        self.save_saliency = args.save_saliency
        self.saliency_outdir = os.path.join(args.outdir, "saliency")
        self.no_expert_memory = args.no_expert_memory
        self.train_episodes = args.train_episodes
        if args.save_saliency and not os.path.exists(self.saliency_outdir):
            os.mkdir(self.saliency_outdir)

    # noinspection PyUnresolvedReferences
    def _create_models(self, args, n_actions, observation_space, model_cls: Type[Union[LinearDQNModel, ConvDQNModel]]):
        if len(observation_space.shape) > 1:
            in_ch, in_w, in_h = observation_space.shape
            in_shape = (in_ch, in_w, in_h)
        else:
            in_shape = None
            in_ch = observation_space.shape[0]

        other_shape = 0
        if isinstance(self.observation_space, TupleSpace):
            other_shape = self.observation_space.other_shape[0]

        # Models
        self.online_net = model_cls(
            args, n_actions, in_ch, cat_in_features=other_shape, in_shape=in_shape
        ).to(self.device)
        self.online_net.train()

        self.target_net = model_cls(
            args, n_actions, in_ch, cat_in_features=other_shape, in_shape=in_shape
        ).to(self.device)
        self.target_net.eval()
        self.update_target_net(step=-1, tau=1.0)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer
        optim_cls = optim.AdamW if args.lambda3 > 0 else optim.Adam
        self.optimizer = optim_cls(params=self.online_net.parameters(), lr=args.lr, weight_decay=args.lambda3)
        if self.test or self.resume:
            self.load(args)
        if self.test:
            self.online_net.eval()

    def explore(self):
        return self.action_space.sample()

    def exploit(self, obs) -> int:
        if self.gen_saliency_maps:
            obs.requires_grad_()
            action = self.online_net.get_action(obs, self.action_space)
            show = input("Show saliency maps?")
            if show == 'y':
                val_s_map, adv_s_maps = self.online_net.get_saliency_maps()
                print(val_s_map)
            return action

        with torch.no_grad():
            action = self.online_net.get_action(obs, self.action_space)
            return action

    def act(self, obs) -> Union[int, OrderedDict]:
        if np.random.rand() < self.logger.epsilon:
            return self.explore()
        if self.noisy:
            self.reset_noise(step=self.logger.step)
        obs = totensor(obs, self.device)
        obs = unsqueeze_obs(obs)
        action = self.exploit(obs)
        return action

    def play_step(self, env, obs, memory: Union[DemoReplayBuffer, None]):
        action = self.act(obs)
        n_obs, reward, done, _ = env.step(action)
        self.logger.append_reward(reward)

        if not self.test:
            memory.append(
                state=obs,
                action=action,
                reward=int(reward),
                done=done,
                next_state=n_obs
            )

            # Train
            if self.logger.step >= self.learn_start:
                self.update_target_net(self.logger.step)
                if not self.greedy:
                    self.logger.update_epsilon(self.learn_start, self.epsilon_final, self.epsilon_steps)
                if self.logger.step % self.replay_frequency == 0:
                    loss, *_ = self.learn(memory=memory)
                    # self.logger.add_expert_percentage(memory.demo_samples / self.batch_size)
                    self.logger.add_loss(loss)
                if self.logger.step % self.save_freq == 0:
                    self.save()
        return n_obs, done

    def train_agent(self, args: Arguments, env, memory: Union[DemoReplayBuffer, PrioritizedExperienceBuffer, None], p_idx=0):
        progressbar = tqdm(
            range(args.train_steps),
            desc='Ep: 0',
            disable=args.verbosity < 1,
            dynamic_ncols=True)

        self.logger.episode = 1
        env.seed(args.seed)
        env.seed(np.random.randint(0, 2**31-1))
        obs = env.reset()

        for self.logger.step in progressbar:
            progressbar.set_description(desc=f'Ep:{self.logger.episode}')
            # self.reset_noise(step=self.logger.step)
            if args.render:
                env.render()
            obs, done = self.play_step(env=env, obs=obs, memory=memory)
            if done:
                self.logger.finish_episode(verbose=args.verbosity > 1)
                env.seed(np.random.randint(0, 2**31-1))
                # if args.fix_seed:
                #     env.seed(np.random.randint(0, 2**31-1))
                # else:
                #     env.seed(np.random.randint(low=0, high=2**31-1))
                obs = env.reset()
                self.logger.episode += 1
                if self.logger.episode > self.train_episodes:
                    break

        progressbar.close()

    def test_agent(self, args: Arguments, env, p_idx=0):
        self.train_agent(args, env, memory=None, p_idx=0)

    def reset_noise(self, step=-1, target=False):
        if (step + 1) % self.replay_frequency == 0:
            model = self.online_net if not target else self.target_net
            model.reset_noise()

    def learn(self, memory: DemoReplayBuffer, p_idx=0):
        forget_percentage = max(self.forget_min, 1.0 - self.logger.step/self.forget_final_step)
        if (not self.no_expert_memory) and (not memory.pretrain_phase) and forget_percentage == 1.0 - self.logger.step/self.forget_final_step:
            print("Reducing weight decay")
            for group in self.optimizer.param_groups:
                group["weight_decay"] = self.lambda3 * forget_percentage
                print(group["weight_decay"])
        samples, is_weights = memory.sample(self.batch_size, demo_fraction=forget_percentage, p_idx=p_idx)
        states, next_states, actions, rewards, non_terminals, \
            is_weights, experts, expert_scales = self._extract_samples(samples, is_weights)

        qvals = self.online_net(states, ret_func=None)
        selected_qvals = self._get_selected_qvals(qvals, actions)

        el_wise_loss = torch.zeros(self.batch_size, device=self.device)
        td_loss = torch.zeros(self.batch_size, device=self.device)

        if self.n_step == 1 or self.dqfd_loss:
            one_loss, one_td_loss = self._calc_loss(selected_qvals, next_states, rewards,
                                                    non_terminals, gamma=self.gamma)
            el_wise_loss += self.lambda0 * one_loss
            td_loss += one_td_loss
        if self.n_step > 1 or self.dqfd_loss:
            nth_states, nth_rewards, nth_non_terminals = self._extract_n_samples(samples)
            n_loss, n_td_loss = self._calc_loss(selected_qvals, nth_states, nth_rewards,
                                                nth_non_terminals, gamma=self.gamma**self.n_step)
            el_wise_loss += self.lambda1 * n_loss
            td_loss += n_td_loss

        # Expert loss
        e_loss, btn_acc, cam_acc = 0, 0, 0
        if torch.sum(experts).item() > 0:
            e_loss, btn_acc, cam_acc = self._bc(qvals, actions, experts)
            # e_loss, btn_acc, cam_acc = self._dqfd(qvals, actions, experts)
            if self.batch_accumulator == 'mean':
                e_loss /= self.nr_action_branches
            e_loss *= expert_scales[experts]
            el_wise_loss[experts] += self.lambda2 * e_loss

        # print(f"TD loss: {el_wise_loss.mean()}")
        # print(f"Expert loss: {e_loss.mean()}")

        loss = (el_wise_loss * is_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()  # Back-propagate importance-weighted mini-batch loss
        # torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        loss_val = torch.abs(td_loss).detach().cpu().numpy()

        memory.update_priorities(loss_val, p_idx=p_idx)
        return np.mean(loss.detach().cpu().numpy()), btn_acc, cam_acc

    def pretrain(
            self, args: Arguments, action_space: Union[spaces.Dict, spaces.Discrete],
            pretrain_action_space: Union[spaces.Dict, spaces.Discrete], memory: DemoReplayBuffer):
        self.batch_size = args.pretrain_batch_size
        # change_optimizer_lr(self.optimizer, args.lr * (args.pretrain_batch_size / args.batch_size))
        memory.set_pretrain_phase(True)
        bar = tqdm(
            range(0, args.pretrain_steps),
            desc=f'Step: 0',
            dynamic_ncols=True, leave=False,
            disable=args.verbosity < 1
        )
        loss_arr = []
        btn_acc_arr = []
        cam_acc_arr = []
        for step in bar:
            self.update_target_net(step)
            if (step + 1) % args.save_freq == 0:
                self.save()
            bar.set_description(desc=f'Step: {step}')
            loss, btn_acc, cam_acc = self.learn(memory=memory)
            loss_arr.append(loss)
            btn_acc_arr.append(btn_acc)
            cam_acc_arr.append(cam_acc)
            if step % 250 == 0:
                loss = f"{np.mean(loss_arr):.4f}"
                cam_mean = np.mean(cam_acc_arr) * 100
                btn_mean = np.mean(btn_acc_arr) * 100
                if self.logger.dw is not None:
                    self.logger.dw.write_pretrain_data(step, loss, btn_mean, cam_mean)
                result = str(
                    f'Loss: {loss}; \t'
                    f'Button accuracy: {btn_mean:.4f}; '
                )
                if self.cam_branch_idxs is not None:
                    result = ''.join([result, f'Camera acc: {cam_mean:.4f};  '])
                print_b(result)

                loss_arr.clear()
                btn_acc_arr.clear()
                cam_acc_arr.clear()
                # Threshold values chosen solely for the purpose of speeding up training with minimum number of steps
                if step >= 10000 and cam_mean > 60 and btn_mean > 96:
                    bar.close()
                    break

        self.batch_size = args.batch_size
        memory.set_pretrain_phase(False)
        data = memory.demo_memory.data
        discounted_rewards = mean([
            int("discounted_reward" in sample[0]) for sample in data if sample is not None
        ])
        print(f"Memory sampled: {discounted_rewards*100:.2f} %")
        return None

    def _get_next_qvals(self, states) -> Union[torch.Tensor, List[torch.Tensor]]:
        with torch.no_grad():
            self.reset_noise(target=True)
            if not self.double_dqn:
                n_qvals: List[torch.Tensor] = self.target_net(states)
                return torch.cat([n_qval.argmax(1, keepdim=True) for n_qval in n_qvals], dim=0)
            else:
                n_qvals: List[torch.Tensor] = self.online_net(states)
                t_qvals: List[torch.Tensor] = self.target_net(states)
                return torch.cat([
                    t_qval[range(self.batch_size), n_qval.argmax(1)]
                    for t_qval, n_qval in zip(t_qvals, n_qvals)
                ], dim=0)

    def _get_selected_qvals(self, qvals, actions):
        if self.nr_action_branches == 1 and actions.shape[0] == self.batch_size:
            return qvals[0][range(self.batch_size), actions]
        # else:
        return torch.cat([
            qval_branch[range(self.batch_size), act_branch] for qval_branch, act_branch in zip(qvals, actions)
        ], dim=0)

    def _calc_loss(self, selected_qvals, next_states, rewards, non_terminals, gamma):
        next_selected_qvals = self._get_next_qvals(next_states)
        el_wise_loss, td_loss = self._calculate_td_loss(
            selected_qvals, next_selected_qvals, rewards, non_terminals, gamma)
        if self.nr_action_branches > 1:
            if self.batch_accumulator == 'mean':
                el_wise_loss = el_wise_loss.view(self.nr_action_branches, self.batch_size).mean(0)
                td_loss = td_loss.view(self.nr_action_branches, self.batch_size).mean(0)
            elif self.batch_accumulator == 'sum':
                el_wise_loss = el_wise_loss.view(self.nr_action_branches, self.batch_size).sum(0)
                td_loss = td_loss.view(self.nr_action_branches, self.batch_size).sum(0)
        return el_wise_loss, td_loss

    def _calculate_td_loss(self, selected_qvals, next_selected_qvals, rewards, non_terminals,
                           gamma) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target_qs = (rewards + non_terminals * gamma * next_selected_qvals.unsqueeze(1)).squeeze()
            td_loss = (selected_qvals - target_qs).detach()
        return F.smooth_l1_loss(selected_qvals, target_qs.squeeze(), reduction='none'), td_loss

    def _bc(self, qvalues, actions, is_experts):
        if not isinstance(qvalues, (List, Tuple)):  # Simple solution for not branching -> assume one branch
            qvalues = [qvalues]
            actions = [actions]
        n_experts = torch.sum(is_experts).item()
        e_loss = torch.zeros(n_experts, device=self.device)
        btn_acc = 0.0
        cam_acc = 0.0
        if torch.any(is_experts):
            for idx, (qs, action) in enumerate(zip(qvalues, actions)):  # Loop branches
                if self.cam_branch_idxs is not None and idx in self.cam_branch_idxs:
                    # bc_loss, accuracy = cam_bc_loss(
                    loss, accuracy = bc_loss(
                        qs, exp_actions=action, is_experts=is_experts
                    )
                    cam_acc += accuracy
                else:
                    loss, accuracy = bc_loss(
                        qs, exp_actions=action, is_experts=is_experts
                    )
                    btn_acc += accuracy
                e_loss += loss

            if self.nr_action_branches > 1:
                btn_acc = btn_acc / (self.nr_action_branches - 2)  # Subtract to account for camera
                cam_acc = cam_acc / 2
        return e_loss, btn_acc, cam_acc

    # Save model parameters on current device (don't move model between devices)
    def save(self):
        save_checkpoint(arch=self.arch, state={
            'arch': self.arch,
            'model_state_dict': self.online_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=True)

    def load(self, args):
        load_model(args, self.arch, self.online_net, self.optimizer)
        self.update_target_net(step=-1, tau=1.0)

    def _extract_samples(self, samples: ExperienceSamples, is_weights):
        states = totensor(samples.states, self.device)
        next_states = totensor(samples.next_states, self.device)

        actions = torch.tensor(samples.actions).to(self.device).long()
        if self.nr_action_branches > 1:
            actions = actions.transpose(0, 1)

        rewards = get_repeated_data(samples.rewards, self.device, (self.nr_action_branches, 1))
        non_terminals = get_repeated_data(samples.non_terminals, self.device, (self.nr_action_branches, 1))

        if self.prioritized:
            is_weights = torch.from_numpy(is_weights).to(self.device)
        else:
            is_weights = torch.ones(self.batch_size, device=self.device)
        expert = torch.tensor(samples.experts, dtype=torch.bool).to(self.device)
        expert_scales = torch.tensor(samples.expert_scales, dtype=torch.float).to(self.device)

        return (
            states, next_states, actions, rewards, non_terminals, is_weights, expert, expert_scales
        )

    def _extract_n_samples(self, samples: ExperienceSamples):
        nth_states = totensor(samples.nth_states, self.device)
        nth_rewards = get_repeated_data(samples.nth_rewards, self.device, (self.nr_action_branches, 1))
        nth_non_terminals = get_repeated_data(samples.nth_non_terminals, self.device, (self.nr_action_branches, 1))
        return nth_states, nth_rewards, nth_non_terminals

    def update_target_net(self, step, tau=1e-2):
        if (step + 1) % self.target_update == 0:
            if self.target_update == 1:
                soft_update(self.online_net, self.target_net, tau)
            else:
                online_state_dict = self.online_net.state_dict()
                correseponding_params = {k: v for k, v in online_state_dict.items()
                                         if k in self.target_net.state_dict()}
                self.target_net.load_state_dict(correseponding_params)


def get_repeated_data(data: np.ndarray, device: torch.device, repeats: Tuple[int, int]):
    data = torch.tensor(data, dtype=torch.float).to(device).unsqueeze(1)
    data = data.repeat(repeats)
    return data


def change_optimizer_lr(optimizer: torch.optim.Optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr
