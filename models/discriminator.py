import torch
from torch import nn
from models import ResNeXtBottleneck, DownBlock, Flatten


class Discriminator(nn.Module):
    def __init__(self, dim: int):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(*[
            DownBlock(3, dim // 2, 4),
            DownBlock(dim // 2, dim // 2, 3),
            DownBlock(dim // 2, dim * 1, 4),
            ResNeXtBottleneck(dim * 1, dim * 1, cardinality=4, dilate=1),
            DownBlock(dim * 1, dim * 1, 3),
            DownBlock(dim * 1, dim * 2, 4),
            ResNeXtBottleneck(dim * 2, dim * 2, cardinality=4, dilate=1),
            DownBlock(dim * 2, dim * 2, 3),
            DownBlock(dim * 2, dim * 4, 4),
            ResNeXtBottleneck(dim * 4, dim * 4, cardinality=4, dilate=1),
            Flatten()
        ])

        self.last = nn.Linear(256 * 8 * 8, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """ Feed forward method of Discriminator

        Args:
            tensor (torch.Tensor): 4D(BCHW) RGB image tensor
        Returns:
            torch.Tensor: [description] 2D(BU) sigmoid output tensor
        """
        tensor = self.main(tensor)
        tensor = self.last(tensor)
        return self.sigmoid(tensor)
