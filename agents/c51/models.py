from collections import OrderedDict
from typing import Union, List, Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gym import spaces

from utils.gen_args import Arguments
from agents.dqn.models import DQNModel, AdvLayer, BranchingAdvLayer
from agents.common import NoisyLinear


class C51AdvLayer(AdvLayer):
    def __init__(self, in_features, out_features, atoms: int, linear_layer: Union[nn.Linear, NoisyLinear]):
        super(C51AdvLayer, self).__init__(in_features, out_features, linear_layer)
        self.adv_layer = nn.Sequential(
            linear_layer(in_features=in_features, out_features=256),
            nn.ReLU(),
            linear_layer(in_features=256, out_features=out_features*atoms),
            FixedReshape(out_features, atoms)
        )


class C51BranchingAdvLayer(BranchingAdvLayer):
    def __init__(self, in_features, n_actions: List[int], atoms: int,
                 linear_layer: Union[nn.Linear, NoisyLinear]):
        super(C51BranchingAdvLayer, self).__init__(in_features, n_actions, linear_layer)
        if type(self) == C51BranchingAdvLayer:
            self.adv_layers = nn.ModuleList([
                C51AdvLayer(in_features, branch_out, atoms, linear_layer) for branch_out in n_actions
            ])


class FixedReshape(nn.Module):
    def __init__(self, *args):
        super(FixedReshape, self).__init__()
        self.dims = args
        self._nelements = np.prod(args)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        reshaped_dims = (batch_size,) + self.dims
        assert x.nelement() == int(self._nelements * batch_size)
        return x.reshape(*reshaped_dims)

    def __repr__(self):
        dims = [f"dim{idx}={value}" for idx, value in enumerate(self.dims)]
        dim_str = ', '.join(dims)
        return f"FixedReshape({dim_str})"


class C51Model(DQNModel):
    def __init__(self, args: Arguments, n_actions, in_channels, in_shape, cat_in_features: int = 0):
        super(C51Model, self).__init__(args, n_actions, in_channels, in_shape, args.atoms, cat_in_features)
        self.n_actions = n_actions
        self.atoms = args.atoms

        # By just adding automatic reshaping of the advantage & value layer,
        # we can use parent class forward as is.
        if self.dueling:
            self.fc_v = nn.Sequential(
                *self.fc_v.children(),
                FixedReshape(1, args.atoms)
            )

    def _make_adv_layers(self, args: Arguments, n_actions, linear_layer):
        if not isinstance(n_actions, Iterable):
            n_actions = [n_actions]
        self.fc_a = C51BranchingAdvLayer(self._linear_out_size, n_actions, args.atoms, linear_layer)
        if args.noisy:
            self._noisy_layers.extend(self.fc_a.get_noisy_layers())

    def forward(self, x: torch.Tensor, ret_func=F.softmax, *args, **kwargs) -> torch.Tensor:
        return super().forward(x, ret_func=ret_func, *args, **kwargs)

    def get_action(self, x: torch.Tensor, action_space: Union[spaces.Dict, spaces.Discrete],
                   support: torch.Tensor = None, *args, **kwargs) -> Union[int, OrderedDict]:
        q_values = self.forward(x, *args, **kwargs)
        if self.branched_out:
            return OrderedDict([
                (key, int((q_val * support).sum(2).argmax(1).item()))
                for key, q_val in zip(action_space.spaces.keys(), q_values)
            ])
        else:
            return int((q_values * support).sum(2).argmax(1).item())


if __name__ == '__main__':
    args = Arguments(underscores_to_dashes=True).parse_args(known_only=True)
    in_shape = (4, 84, 84)
    n_actions = (2, 2, 2, 7, 7)
    model = DQNModel(args, n_actions, in_shape[0], in_shape)
    print(model)
    inp = torch.zeros((args.batch_size,) + in_shape)
    out = model(inp)

    if isinstance(model, C51Model):
        if isinstance(n_actions, Iterable):
            for item, acts in zip(out, n_actions):
                assert item.shape == (args.batch_size, acts, args.atoms)
        else:
            assert out.shape == (args.batch_size, n_actions, args.atoms)
    else:
        if isinstance(n_actions, Iterable):
            for item, acts in zip(out, n_actions):
                assert item.shape == (args.batch_size, acts)
        else:
            assert out.shape == (args.batch_size, n_actions)
