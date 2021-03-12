from collections import OrderedDict
from typing import Union, List, Iterable
import torch
from torch import nn
from gym import spaces

from agents.common import CNNModel, NoisyLinear, LinearModel
from utils.gen_args import Arguments


class AdvLayer(nn.Module):
    def __init__(self, in_features, h_features, out_features, linear_layer: Union[nn.Linear, NoisyLinear]):
        super(AdvLayer, self).__init__()
        self.advantages = None
        if type(self) == AdvLayer:
            self.adv_layer = nn.Sequential(
                linear_layer(in_features=in_features, out_features=h_features),
                nn.ReLU(),
                linear_layer(in_features=h_features, out_features=out_features)
            )

    def get_noisy_layers(self):
        noisy_layers = []
        for module in self.adv_layer.children():
            if isinstance(module, NoisyLinear):
                noisy_layers.append(module)
        return noisy_layers

    def forward(self, x, val=None, ret_func=None):
        adv = self.adv_layer(x)
        if val is not None:
            x = val + (adv - adv.mean(dim=1, keepdim=True))
        else:
            x = adv
        self.advantages = adv
        if ret_func is not None:
            return ret_func(x, dim=-1)
        else:
            return x


class BranchingAdvLayer(nn.Module):
    def __init__(self, in_features, h_features, out_features: List[int],
                 linear_layer: Union[nn.Linear, NoisyLinear]):
        self.nr_action_branches = len(out_features)
        super(BranchingAdvLayer, self).__init__()
        if type(self) == BranchingAdvLayer:
            self.adv_layers = nn.ModuleList([
                AdvLayer(in_features, h_features, branch_out, linear_layer) for branch_out in out_features
            ])

    def get_noisy_layers(self):
        noisy_layers = []
        # noinspection PyTypeChecker
        for adv_layer in self.adv_layers:
            noisy_layers.extend(adv_layer.get_noisy_layers())
        return noisy_layers

    def forward(self, x: torch.Tensor, val=None, ret_func=None):
        if self.nr_action_branches > 1 and x.requires_grad:
            x.register_hook(lambda grad: grad / (self.nr_action_branches+1))
        # noinspection PyTypeChecker
        return tuple(
            adv_layer(x, val, ret_func=ret_func) for adv_layer in self.adv_layers
        )

    def get_advantages(self, idx):
        return self.adv_layers[idx].advantages


class ConvDQNModel(CNNModel):
    def __init__(self, args: Arguments, n_actions, in_channels, in_shape, val_shape=1, cat_in_features: int = 0):
        super(ConvDQNModel, self).__init__(args, in_channels, in_shape, cat_in_features)
        assert len(args.nn_hidden_layers) == 3
        hidden_size = args.nn_hidden_layers[-1]
        self.noisy = args.noisy
        self.dueling = args.dueling
        self.branched_out = isinstance(n_actions, List) and len(n_actions) > 1
        self.values = None

        linear_layer = nn.Linear if not args.noisy else NoisyLinear

        if args.dueling:
            self.fc_v = nn.Sequential(
                linear_layer(self._linear_out_size, hidden_size),
                nn.ReLU(),
                linear_layer(hidden_size, val_shape)
            )
            if args.noisy:
                self._add_noisy_layers(self.fc_v)

        self._make_adv_layers(args, n_actions, linear_layer)

    def _make_adv_layers(self, args, n_actions, linear_layer):
        if not isinstance(n_actions, Iterable):
            n_actions = [n_actions]
        self.fc_a = BranchingAdvLayer(self._linear_out_size, args.nn_hidden_layers[-1], n_actions, linear_layer)

        if args.noisy:
            # noinspection PyUnresolvedReferences
            self._noisy_layers.extend(self.fc_a.get_noisy_layers())

    def reset_noise(self):
        if self.noisy:
            for noisy_layer in self._noisy_layers:
                assert isinstance(noisy_layer, NoisyLinear)
                # noinspection PyUnresolvedReferences
                noisy_layer.reset_noise()

    def forward(self, x: torch.Tensor, ret_func=None, *args, **kwargs) -> torch.Tensor:
        x = super().forward(x, *args, **kwargs)
        val = None
        if self.dueling:
            val = self.fc_v(x)
            self.values = val
        return self.fc_a(x, val, ret_func=ret_func)

    def get_action(self, x: torch.Tensor, action_space: Union[spaces.Dict, spaces.Discrete],
                   *args, **kwargs) -> Union[int, OrderedDict]:
        q_values = self.forward(x, *args, **kwargs)
        if self.branched_out:
            return OrderedDict([
                (key, int(q_val.argmax(1).item())) for key, q_val in zip(action_space.spaces.keys(), q_values)
            ])
        else:
            return int(q_values[0].argmax(1).item())

    def get_values_advantages(self, x: torch.Tensor, action_names: Union[spaces.Dict]):
        self(x)
        if self.values is not None:
            # Values
            yield (x, self.values, "Values", ["Values"])

            # Advantages
            for idx, act_name in enumerate(action_names.spaces.keys()):
                x = x.detach()
                x.requires_grad_()
                self(x)
                advantages = self.fc_a.get_advantages(idx)
                yield (
                    x,
                    advantages,
                    f"{act_name} advantages",
                    list(f"{act_name} {idx}" for idx in range(advantages.shape[1]))
                )


class LinearDQNModel(LinearModel):
    def __init__(self, args: Arguments, n_actions, in_features, val_shape=1, cat_in_features: int = 0,
                 *arg, **kwargs):
        super(LinearDQNModel, self).__init__(args, in_features, cat_in_features=cat_in_features)
        super(LinearDQNModel, self)._create_layers(args, in_features)
        assert len(args.nn_hidden_layers) == 3
        hidden_size = args.nn_hidden_layers[-1]
        self.noisy = args.noisy
        self.dueling = args.dueling
        self.branched_out = isinstance(n_actions, List) and len(n_actions) > 1
        self.values = None

        linear_layer = nn.Linear if not args.noisy else NoisyLinear

        if args.dueling:
            self.fc_v = nn.Sequential(
                linear_layer(self._linear_out_size, hidden_size),
                nn.ReLU(),
                linear_layer(hidden_size, val_shape)
            )
            if args.noisy:
                self._add_noisy_layers(self.fc_v)

        self._make_adv_layers(args, n_actions, linear_layer)

    def _make_adv_layers(self, args, n_actions, linear_layer):
        if not isinstance(n_actions, Iterable):
            n_actions = [n_actions]
        self.fc_a = BranchingAdvLayer(self._linear_out_size, args.nn_hidden_layers[-1], n_actions, linear_layer)

        if args.noisy:
            # noinspection PyUnresolvedReferences
            self._noisy_layers.extend(self.fc_a.get_noisy_layers())

    def reset_noise(self):
        if self.noisy:
            for noisy_layer in self._noisy_layers:
                assert isinstance(noisy_layer, NoisyLinear)
                # noinspection PyUnresolvedReferences
                noisy_layer.reset_noise()

    def forward(self, x: torch.Tensor, ret_func=None, *args, **kwargs) -> torch.Tensor:
        x = super().forward(x, *args, **kwargs)
        val = None
        if self.dueling:
            val = self.fc_v(x)
            self.values = val
        return self.fc_a(x, val, ret_func=ret_func)

    def get_action(self, x: torch.Tensor, action_space: Union[spaces.Dict, spaces.Discrete],
                   *args, **kwargs) -> Union[int, OrderedDict]:
        q_values = self.forward(x, *args, **kwargs)
        if self.branched_out:
            return OrderedDict([
                (key, int(q_val.argmax(1).item())) for key, q_val in zip(action_space.spaces.keys(), q_values)
            ])
        else:
            return int(q_values[0].argmax(1).item())

    def get_values_advantages(self, x: torch.Tensor, action_names: Union[spaces.Dict]):
        self(x)
        if self.values is not None:
            # Values
            yield x, self.values, "Values", ["Values"]
            # Advantages
            for idx, act_name in enumerate(action_names.spaces.keys()):
                x = x.detach()
                x.requires_grad_()
                self(x)
                advantages = self.fc_a.get_advantages(idx)
                yield (
                    x,
                    advantages,
                    f"{act_name} advantages",
                    list(f"{act_name} {idx}" for idx in range(advantages.shape[1]))
                )
