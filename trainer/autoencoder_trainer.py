import os
import time

from tqdm import tqdm
import yaml

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid

from models import AutoEncoder
from data import create_data_loader


class AutoEncoderTrainer:
    def __init__(self, hp: dict, logger: any):
        self.logger = logger
        self.hp = hp
        seed = hp['seed']

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

        self.device = torch.device(
            hp['device'] if torch.cuda.is_available()else 'cpu')
