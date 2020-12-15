import os
import time

from tqdm import tqdm
import yaml

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from torch.utils import tensorboard

from models import AutoEncoder
from trainer import opt
from data import create_data_loader


class AutoEncoderTrainer(opt.TrainerBase):
    def __init__(self, hp: dict, model_name='AutoEncoder'):
        super(AutoEncoderTrainer, self).__init__(hp, model_name)

        hyperparameters = hp['autoencoder']
        self.__model = AutoEncoder(
            dim=hyperparameters['gf_dim'])
        self.__model = self.__model.to(self.device)

        self.__optimizer = optim.Adam(
            self.__model.parameters(),
            hyperparameters['lr'],
            betas=(
                hyperparameters['beta1'],
                hyperparameters['beta2']
            ))

        self.__opti_scheduler = optim.lr_scheduler.MultiStepLR(
            self.__optimizer,
            milestones=[hyperparameters['lr_milestones']],
            gamma=0.1
        )

        self.__l1_loss = nn.L1Loss()
        self.__l1_loss = self.__l1_loss.to(self.device)

        try:
            ckpt = torch.load(hyperparameters['ckpt'])
            self.epoch = ckpt['epoch']
            self.itr = ckpt['itr']
            self.__model.load_state_dict(ckpt['AutoEncoder'])
            self.__optimizer.load_state_dict(ckpt['adam'])
            self.__opti_scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            pass
        finally:
            print("AutoEncoder Trainer Init Done")

    def train(self):
        """ train methods  """

        train_set, test_set = create_data_loader(self.hp, 'autoencoder')
        batch = next(iter(test_set))
        image, line = [data.to(self.device) for data in batch]
        sample_batch = (image, line)
        hyperparametsers = self.hp['autoencoder']

        while self.epoch < hyperparametsers['epoch']:
            p_bar = tqdm(train_set, total=len(train_set))
            for batch in p_bar:
                loss = self._train_step(batch)

                if self.itr % hyperparametsers['sampling_interval'] == 0:
                    self._test_step(sample_batch)

                msg = 'E:%d, Itr:%d, Loss:%0.4f' % (
                    self.epoch + 1, self.itr, loss)
                p_bar.set_description(msg)
                self.itr += 1

            self._check_point()
            self.__opti_scheduler.step()
            self.epoch += 1

        """ Model save as torch script """
        file_name = os.path.join(self.tb.log_dir, 'torch_script')
        file_name = os.path.join(file_name, 'AutoEncoder_ts.zip')
        ts_model = torch.jit.script(self.__model.cpu(),
                                    torch.rand([1, 3, 128, 128]))
        ts_model.save(file_name)

    def _train_step(self, batch: tuple) -> float:
        image, line = [data.to(self.device) for data in batch]

        fake_line = self.__model(image)
        l1_loss = self.__l1_loss(fake_line, line)

        self.__model.zero_grad()
        l1_loss.backward()
        self.__optimizer.step()

        if self.itr % self.hp['autoencoder']['log_interval'] == 0:
            self.tb.add_scalar('TRAINING/L1_loss', l1_loss.item(), self.itr)
            self.tb.add_scalar('Learning Rate',
                               self.__opti_scheduler.get_last_lr()[0],
                               self.itr)

        if self.itr % self.hp['autoencoder']['sampling_interval'] == 0:
            log_image = [make_grid(image, image.size(0),
                                   0, range=(-1, 1), normalize=True),
                         make_grid(fake_line, image.size(0),
                                   0, range=(-1, 1), normalize=True),
                         make_grid(line, image.size(0),
                                   0, range=(-1, 1), normalize=True)]

            self.tb.add_image('TRAINING/SampleImage',
                              make_grid(log_image, 1, 0),
                              self.itr)
        return l1_loss.item()

    @torch.no_grad()
    def _test_step(self, batch: tuple):
        """ Test step
            this section's tensor not need to trace gradient

        Args:
            batch (tuple): batch data tuple (image, line)
        """

        self.__model.eval()
        image, line = [data for data in batch]

        fake_line = self.__model.forward(image)
        l1_loss = self.__l1_loss(fake_line, line)

        pix_range = (-1, 1)
        image = make_grid(image, image.size(0),
                          0, True, range=pix_range)
        line = make_grid(line, line.size(0),
                         0, True, range=pix_range)
        fake_line = make_grid(fake_line, fake_line.size(0),
                              0, True, range=pix_range)

        sample_image = make_grid([image, fake_line, line], 1, 0, range=(0, 1))

        file_name = 'sample_image_GS:%s_Loss:%0.4f.jpg' % (
            self.itr, l1_loss.item())

        file_name = os.path.join(self.tb.log_dir,
                                 'image',
                                 file_name)

        save_image(sample_image, file_name)

        self.__model.train(True)

    def _check_point(self):
        """ Save Checkpoint objects
        checkpoint objects contain epoch, itr, model, optimizer, scheduler """
        file_name = os.path.join(self.tb.log_dir,
                                 'ckpt')
        file_name = os.path.join(
            file_name,
            'AutoEncoder_E:%d_GS:%d.pth' % (self.epoch, self.itr))

        ckpt = {'epoch': self.epoch,
                'itr': self.itr,
                'AutoEncoder': self.__model,
                'adam': self.__optimizer,
                'scheduler': self.__opti_scheduler}
        torch.save(ckpt, file_name)
