import os
import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.utils import tensorboard
from torchvision import models


class TrainerBase(metaclass=ABCMeta):
    """ Abstract Class for Trainer """

    def __init__(self,
                 hp: dict,
                 model_name: str):

        self.epoch = 0
        self.itr = 0
        self.hp = hp
        seed = hp['seed']

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        self.device = torch.device(
            hp['device'] if torch.cuda.is_available()else 'cpu')

        start_time = time.strftime(
            '%Y-%m-%d-%H:00',
            time.localtime(time.time()))
        logdir = os.path.join(
            hp['logdir'], model_name, start_time)
        self.tb = tensorboard.SummaryWriter(logdir)
        os.makedirs(self.tb.log_dir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'image'),
                    exist_ok=True)
        os.makedirs(os.path.join(logdir, 'ckpt'),
                    exist_ok=True)
        os.makedirs(os.path.join(logdir, 'torch_script'),
                    exist_ok=True)

    @abstractmethod
    def train(self):
        pass


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self._loss = nn.BCELoss()

    def __call__(self, inputs: torch.Tensor,
                 target_is_real: bool) -> torch.Tensor:
        """ 
        Args:
            inputs (torch.Tensor): Tensor from Discriminator
            target_is_real (bool): bool flag

        Returns:
            torch.Tensor: BCE Loss """

        target_tensor = self.real_label \
            if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(inputs)
        return self._loss(inputs, target_tensor)


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.features = nn.Sequential(*list(vgg16.features.children())[:9])
        self.__model = vgg16.features
        self.__model.eval()

        self.register_buffer('mean', torch.FloatTensor(
            [0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for p in self.__model.parameters():
            p.requires_grad = False

        self.__loss = torch.nn.MSELoss()

    def __pred(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.__model((tensor * 0.5 - self.mean) / self.std)

    def forward(self,
                fake: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """ Calculate Content loss 
        MSE of VGG16 model's feature map between fake, target

        Args:
            fake (torch.Tensor): 4D Tensor of generated Image
            target (torch.Tensor): 4D Tensor of real dataset Image

        Returns:
            torch.Tensor: Content loss (MSE loss)
        """
        with torch.no_grad():
            _fake = self.__pred(fake)
            _target = self.__pred(target)

        loss = self.__loss(_fake, _target)
        return loss
