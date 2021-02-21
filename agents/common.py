from typing import Union, List, Tuple
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.gen_args import Arguments
from memory.common import ReplayBufferAbstract
from agents.resnet_head import IMPALAResnet


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.register_buffer('noise_in', torch.zeros(in_features))
        self.register_buffer('noise_out', torch.zeros(out_features))

        # Noise parameters (not learnable)
        self.register_buffer("epsilon_w", torch.empty((out_features, in_features)))
        self.register_buffer("epsilon_b", torch.empty(out_features))

        # Mean parameters (learnable)
        self.register_parameter("mu_w", nn.Parameter(torch.empty((out_features, in_features))))
        self.register_parameter("mu_b", nn.Parameter(torch.empty(out_features)))

        # Sigma parameters (learnable)
        self.register_parameter("sigma_w", nn.Parameter(torch.empty((out_features, in_features))))
        self.register_parameter("sigma_b", nn.Parameter(torch.empty(out_features)))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        u_bound = 1 / math.sqrt(self.in_features)
        self.mu_w.data.uniform_(-u_bound, u_bound)
        self.mu_b.data.uniform_(-u_bound, u_bound)
        self.sigma_w.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.sigma_b.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, eps: torch.Tensor):
        eps = eps.normal_()
        return eps.sign().mul_(eps.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.noise_in)
        epsilon_out = self._scale_noise(self.noise_out)
        self.epsilon_w.copy_(epsilon_out.ger(epsilon_in))
        self.epsilon_b.copy_(epsilon_out)

    def forward(self, x: torch.Tensor):
        if self.training:
            w = self.mu_w + self.sigma_w * self.epsilon_w
            b = self.mu_b + self.sigma_b * self.epsilon_b
            return F.linear(x, weight=w, bias=b)
        else:
            return F.linear(x, self.mu_w, self.mu_b)

    def __repr__(self):
        return f"NoisyLinear({self.in_features}, {self.out_features}, std_init={self.std_init})"


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.activation = activation()

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.reshape(shape=(x.shape[0], -1))


def nature_cnn(in_channels, in_shape):
    cnn_model = nn.Sequential(
        BasicConv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
        BasicConv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        BasicConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        Flatten()
    )
    test_input = torch.zeros(in_shape).unsqueeze(0)
    test_out = cnn_model(test_input).cpu().detach().numpy().shape
    out_size = int(np.prod(test_out))
    return cnn_model, out_size


def impala_resnet_head(in_channels, in_shape):
    cnn_model = IMPALAResnet(in_channels)
    test_input = torch.zeros(in_shape).unsqueeze(0)
    test_out = cnn_model(test_input).cpu().detach().numpy().shape
    out_size = int(np.prod(test_out))
    return cnn_model, out_size


class LinearModel(nn.Module):
    def __init__(self, args: Arguments, in_features: int = None, out_features=12,
                 cat_in_features: int = 0, hidden_layer=12):
        super(LinearModel, self).__init__()
        self._linear_out_size = out_features
        self._h1_layer_size = hidden_layer
        self._cat_in_features = cat_in_features

        # List to keep references to all Noisy layers, used for reset noise
        self._noisy_layers: List[NoisyLinear] = []
        if type(self) == LinearModel:
            self._create_layers(args, in_features)

    def _create_layers(self, args: Arguments, in_features):
        assert in_features is not None
        linear_layer = nn.Linear if not args.noisy else NoisyLinear
        self.linear = nn.Sequential(
            linear_layer(in_features + self._cat_in_features, self._h1_layer_size),
            nn.ReLU(),
            linear_layer(self._h1_layer_size, self._linear_out_size),
            nn.ReLU()
        )
        if self._cat_in_features > 0:
            in_features = self._cat_in_features
            out_features = 2**round(math.log(in_features, 2))  # Find nearest power of 2
            self.cat_layer = linear_layer(in_features, out_features)
            if args.noisy:
                self._add_noisy_layers(self.cat_layer)

        if args.noisy:
            self._add_noisy_layers(self.linear)

    def _add_noisy_layers(self, layer: Union[nn.Sequential, nn.Module]):
        if isinstance(layer, nn.Sequential):
            for module in layer.children():
                self._add_noisy_layers(module)
        elif isinstance(layer, NoisyLinear):
            self._noisy_layers.append(layer)

    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        if self._cat_in_features > 0:
            assert isinstance(x, Tuple)
            features, additional_input = x
            additional_features = self.cat_layer(additional_input)
            x = torch.cat([features, additional_features], dim=1)
        return self.linear(x)


class CNNModel(LinearModel):
    def __init__(self, args: Arguments, in_channels, in_shape, cat_in_features: int = 0):
        super(CNNModel, self).__init__(args, cat_in_features=cat_in_features)
        self.cnn, self._cnn_out_size = nature_cnn(in_channels, in_shape)
        # self.cnn, self._cnn_out_size = impala_resnet_head(in_channels, in_shape)
        super()._create_layers(args, self._cnn_out_size)
        
    def forward(self, x, *args, **kwargs) -> torch.Tensor:
        if self._cat_in_features > 0:
            assert isinstance(x, Tuple)
            x, additional_input = x
            x = self.cnn(x)
            x = x, additional_input
        else:
            x = self.cnn(x)
        return super().forward(x)


def add_margin_loss(
        q_vals: torch.Tensor,
        exp_actions: torch.Tensor,
        is_experts: torch.Tensor,
        margin_loss,
        support: torch.Tensor = None):
    exp_batch_size = torch.sum(is_experts).item()
    q_vals = q_vals[is_experts]
    with torch.no_grad():
        exp_actions = exp_actions[is_experts]
        adj_qvals = q_vals.new_full(q_vals.size(), margin_loss)
        adj_qvals[range(exp_batch_size), exp_actions] = 0.0
        adj_qvals += q_vals
        # Get best actions after adjustments
        adj_qvals_max, max_indices = adj_qvals.max(1)
    expert_qvals = q_vals[range(exp_batch_size), exp_actions]
    # margin_loss_elwise = F.smooth_l1_loss(adj_qvals_max, expert_qvals, reduction='none')
    margin_loss_elwise = adj_qvals_max.detach() - expert_qvals
    return margin_loss_elwise, torch.mean((margin_loss_elwise == 0.0).float()).detach().cpu().numpy()


def bc_loss(
        q_vals: torch.Tensor,
        exp_actions: torch.Tensor,
        is_experts: torch.Tensor):
    q_vals = q_vals[is_experts]
    exp_actions = exp_actions[is_experts]
    el_wise_loss = F.cross_entropy(q_vals, exp_actions, reduction='none')
    max_indices = q_vals.max(1)[1]
    accuracy = torch.mean((max_indices == exp_actions).float()).detach().cpu().numpy()
    return el_wise_loss, accuracy


def c51_qtarget(
        q_vals: torch.Tensor,
        pred_qvals: torch.Tensor,
        exp_actions: torch.Tensor,
        is_experts: torch.Tensor,
        exp_rewards: torch.Tensor):
    q_vals = q_vals[is_experts]
    exp_actions = exp_actions[is_experts]
    with torch.no_grad():
        q_targets = torch.zeros_like(q_vals)
        q_targets[..., 0] = 1.0
        exp_rewards = exp_rewards.ceil().long()
        q_targets[exp_actions, exp_rewards] = 1.0
    el_wise_loss = F.smooth_l1_loss(q_vals, q_targets, reduction='none').sum(2).mean(1)
    max_indices = pred_qvals.max(1)[1]
    accuracy = torch.mean((max_indices == exp_actions).float()).detach().cpu().numpy()
    return el_wise_loss, accuracy


class AgentAbstract(ABC):
    online_net: CNNModel

    @abstractmethod
    def train_agent(self, args: Arguments, env, memory: ReplayBufferAbstract, p_idx=0):
        pass

    @abstractmethod
    def test_agent(self, args: Arguments, env, p_idx=0):
        pass


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0.01)
