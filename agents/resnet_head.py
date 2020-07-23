from typing import Tuple

import torch
import torch.nn as nn


class InnerBlock(nn.Module):
    def __init__(self, num_channels, batchnorm=False):
        super(InnerBlock, self).__init__()
        layers = [
            nn.ReLU(), nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(num_channels))
        layers.extend([
            nn.ReLU(), nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x1 = self.layers(x)
        x = x + x1
        return x


class Block(nn.Module):
    def __init__(self, in_channels, num_channels, num_blocks, batchnorm=False):
        super(Block, self).__init__()
        self.in_cnn = nn.Conv2d(in_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.blocks = nn.Sequential(
            *[InnerBlock(num_channels, batchnorm=batchnorm) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.in_cnn(x)
        x = self.maxpool(x)
        return self.blocks(x)


class IMPALAResnet(nn.Module):
    def __init__(self, in_channels, batchnorm=False):
        super(IMPALAResnet, self).__init__()
        res_blocks = []
        for num_in_channels, num_out_channels, num_blocks in [[in_channels, 16, 2], [16, 32, 2], [32, 32, 2]]:
            res_blocks.append(
                Block(num_in_channels, num_out_channels, num_blocks, batchnorm=batchnorm)
            )
        self.resnet_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        batch_size = x.shape[0]
        return self.resnet_blocks(x).reshape(batch_size, -1)


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()

    model = IMPALAResnet(12)
    writer.add_graph(model, input_to_model=torch.zeros((1, 12, 64, 64)))
    writer.close()
