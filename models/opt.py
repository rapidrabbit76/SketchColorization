import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class DownBlock(nn.Module):
    """ Simple Convolution Block Conv->activation(leaky relu) """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(DownBlock, self).__init__()

        stride = 1 if kernel_size == 3 else 2
        self.convolution = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=1,
                                     bias=False)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.convolution(tensor)
        return self.act(tensor)


class UpBlock(nn.Module):
    """ Up Convolution Block Using Pixelshuffle
    conv->act-> resnextblocks -> conv -> pixelshuffle -> act """

    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()

        resnext = nn.Sequential(
            *[ResNeXtBottleneck(out_channels,
                                out_channels,
                                cardinality=32,
                                dilate=1) for _ in range(10)])

        self.block = nn.Sequential(nn.Conv2d(in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False),
                                   nn.LeakyReLU(0.2, True),
                                   resnext,
                                   nn.Conv2d(out_channels,
                                             out_channels // 2 * 4,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False),
                                   nn.PixelShuffle(2),
                                   nn.LeakyReLU(0.2, True))

    def forward(self, inputs: torch.Tensor):
        return self.block(inputs)


class ResNeXtBottleneck(nn.Module):
    """ ResNext :
        (Aggregated Residual Transformations for Deep Neural Networks) """

    def __init__(self,
                 in_channels: int = 256,
                 out_channels: int = 256,
                 stride: int = 1,
                 cardinality: int = 32,
                 dilate: int = 1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2

        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=False)

        self.conv_conv = nn.Conv2d(D, D,
                                   kernel_size=2 + stride,
                                   stride=stride,
                                   padding=dilate,
                                   dilation=dilate,
                                   groups=cardinality,
                                   bias=False)

        self.conv_expand = nn.Conv2d(
            D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_expand.forward(bottleneck)
        x = self.shortcut.forward(x)
        return x + bottleneck


class Flatten(nn.Module):
    """ Flatten Layer """

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: 4D Tensor
        :return: 4D Tensor
        """
        return tensor.view(tensor.size(0), -1)


def kaiming_normal(module):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """

    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight.data, a=0.2)
    elif isinstance(module, nn.ConvTranspose2d):
        init.kaiming_normal_(module.weight.data, a=0.2)
    elif isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight.data, a=0.2)


def xavier_normal(module):
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    if isinstance(module, nn.Conv2d):
        init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.ConvTranspose2d):
        init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight.data)
