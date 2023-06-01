import math
from typing import List, Optional, Any

import torch
from torch import nn, Tensor

activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'leaky_relu': nn.LeakyReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'linear': nn.Identity
}


class BasicConv1d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: Optional[str],
            **kwargs: Any
    ) -> None:
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        # if activation is not None:
        #     self.activation = activations[activation]# (inplace=True)
        # else:
        #     self.activation = nn.Identity()
        self.activation = activations[activation]()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x: List[Tensor]):
        return torch.cat(x, dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GAP1d(nn.Module):
    """Global Adaptive Pooling + Flatten
    """

    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Flatten()

    def forward(self, x):
        return self.flatten(self.gap(x))


class Inception(nn.Module):
    """Inception Module
    """

    def __init__(self, n_inputs: int, n_convolutions: int = 3, n_filters: int = 32, kernel_size: int = 39,
                 use_bottleneck: bool = True, bottleneck_size: int = 32, activation: str = 'linear'):
        super(Inception, self).__init__()

        if n_convolutions > math.log2(kernel_size):
            raise AttributeError
        stride = 1  # hard coded, since SAME padding does not support any stride values other than 1

        self.kernel_size = kernel_size

        # self.activation = activations[activation]
        self.bottleneck = nn.Conv1d(in_channels=n_inputs, out_channels=bottleneck_size,
                                    kernel_size=1,
                                    padding='same',
                                    stride=stride,
                                    bias=False) if use_bottleneck else nn.Identity()
        self.conv_layers = nn.ModuleDict()
        for i in range(n_convolutions):
            kernel_size_ = self.kernel_size // (2 ** i)
            # kernel_size_ = self.kernel_size - (2 * i)
            if kernel_size_ % 2 == 0:
                kernel_size_ -= 1

            self.conv_layers[f"Conv1D_{i}"] = BasicConv1d(in_channels=bottleneck_size if use_bottleneck else n_inputs,
                                                          out_channels=n_filters,
                                                          activation=activation,
                                                          kernel_size=kernel_size_,
                                                          padding='same',
                                                          stride=stride)

        self.maxpoolconv = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=stride, padding=1),
                                         nn.Conv1d(in_channels=n_inputs,
                                                   out_channels=n_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   stride=stride,
                                                   bias=False))
        self.concat = Concat()
        self.bn = nn.BatchNorm1d(n_filters * (n_convolutions + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        out = self.concat([l(x) for l in self.conv_layers.values()] + [self.maxpoolconv(input_tensor)])
        out = self.bn(out)
        out = self.relu(out)
        return out


class ShortCut(nn.Module):
    """Skip connection
    """

    def __init__(self, in_channels: int, n_filters: int):
        super(ShortCut, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        out = torch.add(x, y)
        out = self.activation(out)
        return out


class InceptionTime(nn.Module):
    """InceptionTime model definition
    """

    def __init__(self, n_inputs: Optional[int] = None, n_classes: Optional[int] = None, use_residual=True,
                 use_bottleneck=True, depth=6, n_convolutions: int = 3, n_filters: int = 32, kernel_size=32,
                 initialization="kaiming_uniform", **kwargs):
        super(InceptionTime, self).__init__()

        self.blocks = nn.ModuleList([nn.Sequential()])

        for d in range(depth):
            self.blocks[-1].add_module(f"Inception_{d}",
                                       Inception(n_inputs=n_inputs if d == 0 else n_filters * (n_convolutions + 1),
                                                 use_bottleneck=use_bottleneck,
                                                 n_convolutions=n_convolutions,
                                                 n_filters=n_filters,
                                                 kernel_size=kernel_size,
                                                 **kwargs)
                                       )
            if use_residual and d % 3 == 2:
                n_in, n_out = n_inputs if d == 2 else n_filters * (
                        n_convolutions + 1), n_filters * (n_convolutions + 1)
                self.blocks.append(ShortCut(n_in, n_out))
                # n_filters=self.blocks[-1].get_submodule(f"Inception_{d - 1}").maxpoolconv[-1].out_channels * (n_convolutions + 1)))
                if d < depth - 1:
                    self.blocks.append(nn.Sequential())
        self.gap = GAP1d(1)
        self.fc = nn.Linear(n_filters * (n_convolutions + 1), out_features=n_classes, bias=True)
        # self.activation = nn.Softmax(dim=1)

        # for d in range(depth):
        #     self.blocks[-1].add_module(f"Inception_{d}",
        #                                Inception(n_inputs=n_inputs if d == 0 else n_filters * (n_convolutions + 1),
        #                                          use_bottleneck=use_bottleneck, n_convolutions=n_convolutions,
        #                                          n_filters=n_filters, **kwargs))
        #     if use_residual and d % 3 == 2:
        #         n_in, n_out = n_inputs if d == 2 else n_filters * (n_convolutions + 1), n_filters * (n_convolutions + 1)
        #         self.blocks.append(ShortCut(n_in, n_out))
        #         # n_filters=self.blocks[-1].get_submodule(f"Inception_{d - 1}").maxpoolconv[-1].out_channels * (n_convolutions + 1)))
        #         if d < depth - 1:
        #             self.blocks.append(nn.Sequential())
        # self.gap = GAP1d(1)
        # self.fc = nn.Linear(n_filters * (n_convolutions + 1), out_features=n_classes, bias=True)
        # self.activation = nn.Softmax(dim=1)
        if initialization == "glorot_uniform":
            self.apply(self.glorot_uniform_initialization)

    def glorot_uniform_initialization(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform(m.weight.data)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight.data)

    def forward(self, x: torch.Tensor):
        block_in = x
        block_out = torch.ones((2,))
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                block_out = block(block_in)
            else:
                block_in = block(block_in, block_out)

        out = self.gap(block_out)
        out = self.fc(out)
        # https://stackoverflow.com/questions/57342987/translating-pytorch-program-into-keras-different-results
        # Logits are calculated by loss function already!!!
        # out = self.activation(out)
        return out
