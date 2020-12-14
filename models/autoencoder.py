import torch
from torch import nn


class Encoder(nn.Module):
    class Block(nn.Module):
        """ Convolution Block Conv,Norm,Activate """

        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: int) -> None:
            super(Encoder.Block, self).__init__()
            stride = 1 if kernel_size == 3 else 2
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=1, bias=False),
                nn.GroupNorm(4, out_channels),
                nn.ReLU())

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.block(tensor)

    def __init__(self, dim: int):
        Block = self.Block
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            Block(3, dim // 2, 3),           # 32 ,3,1
            Block(dim // 2, dim * 1, 4),     # 64 ,4,2
            Block(dim * 1, dim * 1, 3),      # 64 ,3,1
            Block(dim * 1, dim * 2, 4),      # 128,4,2
            Block(dim * 2, dim * 2, 3),      # 128,3,1
            Block(dim * 2, dim * 4, 4),      # 256,4,2
            Block(dim * 4, dim * 4, 3),      # 256,3,1
            Block(dim * 4, dim * 8, 4),      # 512,4,2
            Block(dim * 8, dim * 8, 3),      # 512,3,1
        )

    def forward(self, tensor: torch.Tensor):
        return self.main(tensor)


class Decoder(Encoder):
    class UpBlock(nn.Module):
        """ Convolution Block Conv,Norm,Activate """

        def __init__(self,
                     in_channels: int,
                     out_channels: int,
                     kernel_size: int) -> None:
            super(Decoder.UpBlock, self).__init__()
            stride = 1 if kernel_size == 3 else 2
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=1,
                                   bias=False),
                nn.GroupNorm(4, out_channels),
                nn.ReLU())

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.block(tensor)

    def __init__(self, dim: int) -> None:
        super(Decoder, self).__init__(dim)
        UpBlock = self.UpBlock
        Block = self.Block

        self.main = nn.Sequential(
            UpBlock(dim * 8, dim * 8, 4),
            Block(dim * 8, dim * 4, 3),
            UpBlock(dim * 4, dim * 4, 4),
            Block(dim * 4, dim * 2, 3),
            UpBlock(dim * 2, dim * 2, 4),
            Block(dim * 2, dim * 1, 3),
            UpBlock(dim * 1, dim * 1, 4),
            Block(dim * 1, dim // 2, 3),
        )

        self.last = nn.Sequential(
            nn.Conv2d(dim // 2, 1, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.Tanh())

    def forward(self, tensor: torch.Tensor):
        tensor = self.main(tensor)
        return self.last(tensor)


class AutoEncoder(nn.Module):
    """ 
        AutoEncoder
        input  shape : 3x128x128
        output shape : 1x128x128
    """

    def __init__(self, dim: int):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

    def forward(self, tensor: torch.Tensor):
        encoder = self.encoder(tensor)
        return self.decoder(encoder)

    def weight_init(self):
        """ Model weight init methods """

        def init(m):
            kaiming = nn.init.kaiming_normal_
            if isinstance(m, nn.conv2d):
                kaiming(m.weight.data)
            elif isinstance(m, nn.batchnorm2d):
                nn.init.normal_(m.weight.data, 0., 0.02)
            elif isinstance(m, nn.linear):
                kaiming(m.weight.data)
                m.bias.data.fill_(0.000001)
        self.apply(init)
