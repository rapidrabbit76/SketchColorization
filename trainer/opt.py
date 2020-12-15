import os
import time
from abc import ABCMeta, abstractmethod

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from torch.utils import tensorboard


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
