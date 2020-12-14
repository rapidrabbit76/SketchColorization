import torch
from torch import nn
from models import UpBlock, DownBlock, kaiming_normal, xavier_normal


class Generator(nn.Module):
    """ Generator of draft model and colorization model
        draft model        : in_dim: 5(line(1),hint(4))
        colorization model : in_dim: 4(line(1), draft(3)) """

    def __init__(self, in_dim: int, dim: int, name: str):
        super(Generator, self).__init__(name=name)

        self.e0 = DownBlock(in_dim, dim // 2, 3)       # 128*128*32
        self.e1 = DownBlock(dim // 2, dim * 1, 4)      # 64*64*64

        self.e2 = DownBlock(dim * 1, dim * 1, 3)       # 64*64*63
        self.e3 = DownBlock(dim * 1, dim * 2, 4)       # 32*32*128

        self.e4 = DownBlock(dim * 2, dim * 2, 3)       # 32*32*128
        self.e5 = DownBlock(dim * 2, dim * 4, 4)       # 16*16*256

        self.e6 = DownBlock(dim * 4, dim * 4, 3)       # 16*16*256
        self.e7 = DownBlock(dim * 4, dim * 8, 4)       # 8*8*512

        self.e8 = DownBlock(dim * 8, dim * 8, 3)       # 8*8*512

        self.d8 = UpBlock(dim * 8 * 2, dim * 8)        # 16*16* 256
        self.d6 = UpBlock(dim * 4 * 2, dim * 4)        # 32*32*128
        self.d4 = UpBlock(dim * 2 * 2, dim * 2)        # 64*64*64
        self.d2 = UpBlock(dim * 1 * 2, dim * 1)        # 128*128*32
        self.d1 = DownBlock(dim // 2, dim // 2, 3)     # 128*128*32

        self.d0 = nn.Sequential(                       # 128*128*3
            nn.Conv2d(dim, 3, kernl_size=3,
                      stride=1, padding=1, bias=False),
            nn.Tanh())

        self.__relu_layers = [self.e0, self.e1, self.e2, self.e3,
                              self.e4, self.e5, self.e6, self.e7,
                              self.e8, self.d8, self.d6, self.d4,
                              self.d2, self.d1]

        self.__than_layers = [self.d0]

        def forward(self,
                    line: torch.Tensor,
                    hint: torch.Tensor) -> torch.Tensor:
            """ Feed forward method of Generator(draft/colorizaton model)

            Args:
                line (torch.Tensor): 4D(BCHW) greyscale image tensor
                hint (torch.Tensor): 4D(BCHW) RGBA image tensor 
                in image tensor RGB scale is -1 to 1 and Alpha scale is 0 to 1 
                if colorization model hint is draft tensor.
                draft tensor color space is RGB and scale is -1 to 1.

                draft model        : line, hint shape is N,1,128,128 (line)
                                     and N,4,128,128 (hint)
                colorization model : line, hint shape is N,1,512,512 (line)
                                     and N,3,512,512 (hint)

            Returns:
                torch.Tensor: draft or color image tensor (RGB Color space) """

            inputs = torch.cat([line, hint], 1)
            e0 = self.e0(inputs)
            e2 = self.e2(self.e1(e0))
            e4 = self.e4(self.e3(e2))
            e6 = self.e6(self.e5(e4))
            e7 = self.e7(e6)
            e8 = self.e8(e7)

            tensor = self.d8(torch.cat([e7, e8], 1))
            tensor = self.d6(torch.cat([e6, tensor], 1))
            tensor = self.d4(torch.cat([e4, tensor], 1))
            tensor = self.d2(torch.cat([e2, tensor], 1))
            tensor = self.d1(tensor)

            return self.d0(torch.cat([e0, tensor], 1))

        def weight_init(self):
            for layer in self.__relu_layers:
                layer.apply(kaiming_normal)
            for layer in self.__than_layers:
                layer.apply(xavier_normal)
