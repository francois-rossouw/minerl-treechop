from typing import Tuple, List, Generator

import torch
from torch.nn import functional as F

from agents.c51.models import C51Model
from agents.dqn.dqn_agent import DQN
from agents.common import bc_loss
from memory.demo_memory import DemoReplayBuffer
from utils.gen_args import Arguments
from utils.mylogger import Logger


class C51(DQN):
    def __init__(self, args: Arguments, logger: Logger, action_space, observation_space, p_idx=0, **kwargs):
        """
        C51 agent extended from DQN agent class
        :param args: all default arguments. Can be changed in cmd line
        :param logger: logging object
        :param action_space: action space of gym env
        :param observation_space: observation space of gym env
        :param p_idx: policy index
        :return: self
        """
        super(C51, self).__init__(args, logger, action_space, observation_space, p_idx, **kwargs)

        self.arch = "C51"
        if args.double_dqn:
            self.arch = f"Double{self.arch}"
        if args.dueling:
            self.arch = f"Dueling{self.arch}"

        self.full_batch = self.nr_action_branches * self.batch_size
        self._linear_batch_idxs = torch.arange(self.full_batch).to(self.device).unsqueeze(-1).expand(
            self.full_batch, args.atoms).reshape(-1) * args.atoms

        if type(self) == C51:
            # Only create models if not called from a subclass
            self._create_models(args, self.n_actions, observation_space, C51Model)

    def _set_args(self, args: Arguments):
        super()._set_args(args)
        self.atoms = args.atoms
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.support_atoms = torch.linspace(args.v_min, args.v_max, args.atoms).to(self.device)
        self.delta_z = (args.v_max - args.v_min) / (args.atoms - 1)

    def exploit(self, obs) -> int:
        with torch.no_grad():
            action = self.online_net.get_action(obs, action_space=self.action_space, support=self.support_atoms)
            return action

    def learn(self, memory: DemoReplayBuffer, p_idx=0):
        if self._linear_batch_idxs.nelement() != (self.batch_size*self.nr_action_branches*self.atoms):
            # Adapt for changing batch size (pretrain vs RL)
            self.full_batch = self.nr_action_branches * self.batch_size
            self._linear_batch_idxs = torch.arange(self.full_batch).to(self.device).unsqueeze(-1).expand(
                self.full_batch, self.atoms).reshape(-1) * self.atoms

        return super().learn(memory, p_idx)

    def _get_next_qvals(self, states) -> torch.Tensor:
        with torch.no_grad():
            self.reset_noise(target=True)
            t_qvals: List[torch.Tensor] = self.target_net(states)
            if not self.double_dqn:
                n_qvals: List[torch.Tensor] = t_qvals
            else:
                n_qvals: List[torch.Tensor] = self.online_net(states)
            a_stars: List[torch.Tensor] = [
                (self.support_atoms.expand_as(n_qval) * n_qval).sum(2).argmax(1) for n_qval in n_qvals
            ]
            return torch.cat([
                t_qval[range(self.batch_size), a_star] for t_qval, a_star in zip(t_qvals, a_stars)
            ], dim=0)

    def _calculate_td_loss(self, selected_qvals, next_selected_qvals, rewards, non_terminals,
                           gamma) -> Tuple[torch.Tensor, torch.Tensor]:
        qvals = F.log_softmax(selected_qvals, dim=1)
        with torch.no_grad():
            # Compute the projection of Tz onto the support z
            tz = rewards + non_terminals * gamma * self.support_atoms
            tz = tz.clamp(min=self.v_min, max=self.v_max)  # Clamp between supported values
            b = (tz - self.v_min) / self.delta_z
            l, u = b.floor().long(), b.ceil().long()
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m_l = torch.zeros_like(b)
            m_u = torch.zeros_like(b)
            l_indices = self._linear_batch_idxs + l.view(-1)  # Linear indices (N*atoms + idx)
            u_indices = self._linear_batch_idxs + u.view(-1)  # Linear indices (N*atoms + idx)
            # Place target q-values at indices in 2d array
            m_l.put_(l_indices, next_selected_qvals * (u.float() - b), accumulate=True)
            m_u.put_(u_indices, next_selected_qvals * (b - l.float()), accumulate=True)
            m = m_l + m_u

        loss = -torch.sum(m * qvals, dim=1)
        return loss, loss.detach()

    def _bc(self, qvalues, actions, is_experts):
        if not isinstance(qvalues, (List, Tuple, Generator)):  # Simple solution for not branching -> assume one branch
            qvalues = [qvalues]
            actions = [actions]
        n_experts = torch.sum(is_experts).item()
        e_loss = torch.zeros(n_experts, device=self.device)
        btn_acc = 0.0
        cam_acc = 0.0
        if torch.any(is_experts):
            for idx, (qs, action) in enumerate(zip(qvalues, actions)):  # Loop branches
                qvals = F.softmax(qs, dim=2)
                qvals = (qvals * self.support_atoms).sum(2)

                if self.cam_branch_idxs is not None and idx in self.cam_branch_idxs:
                    # bc_loss, accuracy = cam_bc_loss(
                    loss, accuracy = bc_loss(
                        qvals, exp_actions=action, is_experts=is_experts
                    )
                    cam_acc += accuracy
                else:
                    loss, accuracy = bc_loss(
                        qvals, exp_actions=action, is_experts=is_experts
                    )
                    btn_acc += accuracy
                e_loss += loss
            if self.nr_action_branches > 1:
                btn_acc = btn_acc / (self.nr_action_branches - 2)  # Subtract to account for camera
                cam_acc = cam_acc / 2
        return e_loss, btn_acc, cam_acc
