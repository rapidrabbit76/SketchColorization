import os
import time

from tqdm import tqdm
import yaml

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from torch.utils import tensorboard

from models import Generator, Discriminator
from trainer import opt
from data import create_data_loader


class DraftModelTrainer(opt.TrainerBase):
    def __init__(self, hp: dict, model_name='DraftModel'):
        super(DraftModelTrainer, self).__init__(hp, model_name)

        hyperparameters = hp['draft']

        self.__generator = Generator(hyperparameters['in_dim'],
                                     hyperparameters['gf_dim'])
        self.__generator = self.__generator.to(self.device)

        self.__discriminator = Discriminator(hyperparameters['gf_dim'])
        self.__discriminator = self.__discriminator.to(self.device)

        self.__autoencoder = torch.jit.load(
            hyperparameters['autoencoder_path'])
        self.__autoencoder = self.__autoencoder.to(self.device)
        self.__autoencoder.eval()

        self.__gan_loss = opt.GANLoss().to(self.device)
        self.__content_loss = opt.ContentLoss().to(self.device)
        self.__l1_loss = nn.L1Loss().to(self.device)

        self.__w_gan = hyperparameters['w_gan']
        self.__w_recon = hyperparameters['w_recon']
        self.__w_cont = hyperparameters['w_cont']
        self.__w_line = hyperparameters['w_line']

        self.__optimizer_generator = optim.Adam(
            self.__generator.parameters(),
            lr=hyperparameters['lr'],
            betas=(hyperparameters['beta1'], hyperparameters['beta2']))

        self.__optimizer_discriminator = optim.Adam(
            self.__discriminator.parameters(),
            lr=hyperparameters['lr'],
            betas=(hyperparameters['beta1'], hyperparameters['beta2']))

        self.__opti_gen_scheduler = optim.lr_scheduler.MultiStepLR(
            self.__optimizer_generator,
            milestones=[hyperparameters['lr_milestones']],
            gamma=0.1)

        self.__opti_dis_scheduler = optim.lr_scheduler.MultiStepLR(
            self.__optimizer_discriminator,
            milestones=[hyperparameters['lr_milestones']],
            gamma=0.1)

        try:
            ckpt = torch.load(hyperparameters['ckpt'])
            self.epoch = ckpt['epoch']
            self.itr = ckpt['itr']
            self.__generator.load_state_dict(ckpt['Generator'])
            self.__discriminator.load_state_dict(ckpt['Discriminator'])
            self.__optimizer_generator.load_state_dict(
                ckpt['adma_generator'])
            self.__optimizer_discriminator.load_state_dict(
                ckpt['adma_discriminator'])
            self.__opti_gen_scheduler.load_state_dict(
                ckpt['scheduler_generator'])
            self.__opti_dis_scheduler.load_state_dict(
                ckpt['scheduler_discriminator'])
        except Exception as e:
            pass
        finally:
            print("DratfModel Trainer Init Done")

    def train(self):
        """ train methods  """

        train_set, test_set = create_data_loader(self.hp, 'draft')
        batch = next(iter(test_set))
        target, hint, line = [data.to(self.device) for data in batch]
        sample_batch = (target, hint, line)
        hyperparametsers = self.hp['draft']

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
            self.__opti_dis_scheduler.step()
            self.__opti_gen_scheduler.step()
            self.epoch += 1

        """ Model save as torch script """
        file_name = os.path.join(self.tb.log_dir, 'torch_script')
        file_name = os.path.join(file_name, 'DraftModel_ts.zip')
        ts_model = torch.jit.script(self.__generator.cpu(),
                                    torch.rand([1, 3, 128, 128]))
        ts_model.save(file_name)

    def _train_step(self, batch: tuple) -> float:

        target, hint, line = [data.to(self.device) for data in batch]

        ####################
        #   Discriminator  #
        ####################
        fake_image = self.__generator(line, hint)  # G(l,h)
        fake_dis = self.__discriminator(fake_image.detach())  # D(G(l,h))
        real_dis = self.__discriminator(target)  # D(c)

        fake_loss = self.__gan_loss(fake_dis, False)
        real_loss = self.__gan_loss(real_dis, True)
        discriminator_loss = fake_loss + real_loss

        self.__discriminator.zero_grad()
        discriminator_loss.backward()
        self.__optimizer_discriminator.step()

        ####################
        #     Generator    #
        ####################
        _fake_dis = self.__discriminator(fake_image)

        with torch.no_grad():
            fake_line = self.__autoencoder(fake_image)
            real_line = self.__autoencoder(target)

        adv_loss = self.__gan_loss(_fake_dis, True)
        recon_loss = self.__l1_loss(fake_image, target)
        content_loss = self.__content_loss(fake_image, target)
        line_loss = self.__l1_loss(fake_line, real_line)

        generator_loss = (adv_loss * self.__w_gan) \
            + (recon_loss * self.__w_recon) \
            + (content_loss * self.__w_cont) \
            + (line_loss * self.__w_line)

        self.__generator.zero_grad()
        generator_loss.backward()
        self.__optimizer_generator.step()

        ####################
        #      Logging     #
        ####################
        if self.itr % self.hp['draft']['log_interval'] == 0:
            self.tb.add_scalar('TRAINING/Discriminator.loss',
                               discriminator_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Discriminator.loss.fake',
                               fake_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Discriminator.loss.real',
                               real_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Generator.loss',
                               generator_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Generator.loss.adv',
                               adv_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Generator.loss.recon',
                               recon_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Generator.loss.content',
                               content_loss.item(), self.itr)
            self.tb.add_scalar('TRAINING/Generator.line',
                               line_loss.item(), self.itr)

            self.tb.add_scalar('Learning Rate',
                               self.__opti_dis_scheduler.get_last_lr()[0],
                               self.itr)

        if self.itr % self.hp['draft']['sampling_interval'] == 0:
            r, g, b, a = torch.chunk(hint, 4, 1)
            hint = torch.cat([r, g, b], 1)
            batch_size = target.size(0)

            log_image = [make_grid(target, batch_size, 0,
                                   range=(-1, 1), normalize=True),
                         make_grid(fake_image, batch_size, 0,
                                   range=(-1, 1), normalize=True),
                         make_grid(line, batch_size, 0,
                                   range=(-1, 1), normalize=True),
                         make_grid(hint, batch_size, 0,
                                   range=(-1, 1), normalize=True)]

            self.tb.add_image('TRAINING/SampleImage',
                              make_grid(log_image, 1, 0),
                              self.itr)
        return generator_loss.item()

    @ torch.no_grad()
    def _test_step(self, batch: tuple):
        """ Test step
            this section's tensor not need to trace gradient

        Args:
            batch (tuple): batch data tuple (target, hint, line) """

        self.__generator.eval()
        target, hint, line = [data for data in batch]
        zero_hint = torch.zeros_like(hint)
        fake_image = self.__generator.forward(line, hint)
        fake_zero_hint_image = \
            self.__generator.forward(line, zero_hint)
        fake_line = self.__autoencoder(fake_image)
        real_line = self.__autoencoder(target)

        recon_loss = self.__l1_loss(fake_image, target)
        content_loss = self.__content_loss(fake_image, target)
        line_loss = self.__l1_loss(fake_line, real_line)

        self.tb.add_scalar('TESTING/Generator.loss.recon',
                           recon_loss.item(), self.itr)
        self.tb.add_scalar('TESTING/Generator.loss.content',
                           content_loss.item(), self.itr)
        self.tb.add_scalar('TESTING/Generator.line',
                           line_loss.item(), self.itr)

        r, g, b, a = torch.chunk(hint, 4, 1)
        hint = torch.cat([r, g, b], 1)
        batch_size = target.size(0)
        log_image = [make_grid(target, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(fake_image, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(fake_zero_hint_image, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(line, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(hint, batch_size, 0,
                               range=(-1, 1), normalize=True)]

        sample_image = make_grid(log_image,
                                 1, 0, range=(0, 1))

        self.tb.add_image('TESTING/SampleImage',
                          sample_image,
                          self.itr)

        file_name = 'sample_image_GS:%d.jpg' % self.itr
        file_name = os.path.join(self.tb.log_dir, 'image',
                                 file_name)
        save_image(sample_image, file_name)
        self.__generator.train(True)

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
                'Generator': self.__generator,
                'Discriminator': self.__discriminator,
                'adma_generator': self.__optimizer_generator,
                'adma_discriminator': self.__optimizer_discriminator,
                'scheduler_generator': self.__opti_gen_scheduler,
                'scheduler_discriminator': self.__opti_dis_scheduler}
        torch.save(ckpt, file_name)
