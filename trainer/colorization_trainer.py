import os

from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision.utils import save_image, make_grid

from models import Generator, SketchColorizationModel
from trainer import opt
from data import create_data_loader, DraftArgumentation


class ColorizationModelTrainer(opt.TrainerBase):
    def __init__(self, hp: dict, model_name='ColorizationModel'):
        super(ColorizationModelTrainer, self).__init__(hp, model_name)

        hyperparameters = hp['colorization']

        self.__generator = Generator(hyperparameters['in_dim'],
                                     hyperparameters['gf_dim'])
        self.__generator = self.__generator.to(self.device)

        self.__draft_model = torch.jit.load(
            hyperparameters['draft_model_path']).to(self.device)
        self.__draft_model.eval()

        self.__l1_loss = nn.L1Loss().to(self.device)

        self.__optimizer = optim.Adam(
            self.__generator.parameters(),
            lr=hyperparameters['lr'],
            betas=(hyperparameters['beta1'], hyperparameters['beta2']))

        self.__draft_augmentation = DraftArgumentation(self.device)

        self.__scheduler = optim.lr_scheduler.MultiStepLR(
            self.__optimizer,
            milestones=[hyperparameters['lr_milestones']],
            gamma=0.1)

        try:
            ckpt = torch.load(hyperparameters['ckpt'])
            self.epoch = ckpt['epoch']
            self.itr = ckpt['itr']
            self.__generator.load_state_dict(ckpt['Generator'])
            self.__optimizer.load_state_dict(ckpt['adma'])
            self.__scheduler.load_state_dict(ckpt['scheduler'])
        except Exception:
            pass
        finally:
            print("Colorization Trainer Init Done")

    def train(self):
        """ train methods  """

        train_set, test_set = create_data_loader(self.hp, 'colorization')
        batch = next(iter(test_set))
        target, hint, line, line_draft = [
            data.to(self.device) for data in batch]
        sample_batch = (target, hint, line, line_draft)
        hyperparametsers = self.hp['colorization']

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
            self.__scheduler.step()
            self.epoch += 1

        """ Model save as torch script """
        file_name = os.path.join(self.tb.log_dir, 'torch_script')
        deployment_model_file_name = os.path.join(
            file_name, 'SketchColorizationModel.zip')
        file_name = os.path.join(file_name, 'Colorization_ts.zip')
        ts_model = torch.jit.script(self.__generator.cpu(),
                                    torch.rand([1, 3, 128, 128]))
        ts_model.save(file_name)

        deployment_model = SketchColorizationModel(
            hyperparametsers['gf_dim'])
        deployment_model_ts = torch.jit.script(
            deployment_model)
        deployment_model_ts.draft_model.load_state_dict(
            self.__draft_model.cpu().state_dict())
        deployment_model_ts.colorization_model.load_state_dict(
            ts_model.state_dict())
        deployment_model_ts.save(deployment_model_file_name)

    def _train_step(self, batch: tuple) -> float:
        target, hint, line, line_draft = [
            data.to(self.device) for data in batch]

        #############
        #   Draft   #
        #############
        with torch.no_grad():
            draft = self.__draft_model.forward(line_draft, hint)
            draft = self.__draft_augmentation(draft)
            draft = nn.functional.interpolate(draft, size=512)

        ####################
        #     Generator    #
        ####################

        fake_image = self.__generator(line, draft)
        generator_loss = self.__l1_loss(fake_image, target)

        self.__generator.zero_grad()
        generator_loss.backward()
        self.__optimizer.step()

        ####################
        #      Logging     #
        ####################
        if self.itr % self.hp['colorization']['log_interval'] == 0:
            self.tb.add_scalar('TRAINING/Generator.loss',
                               generator_loss.item(), self.itr)
            self.tb.add_scalar('Learning Rate',
                               self.__scheduler.get_last_lr()[0],
                               self.itr)

        if self.itr % self.hp['colorization']['sampling_interval'] == 0:
            r, g, b, a = torch.chunk(hint, 4, 1)
            hint = torch.cat([r, g, b], 1)
            batch_size = target.size(0)
            hint = nn.functional.interpolate(hint, size=512)

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
        target, hint, line, line_draft = [
            data.to(self.device) for data in batch]
        zero_hint = torch.zeros_like(hint)

        draft = self.__draft_model.forward(line_draft, hint)
        draft = nn.functional.interpolate(draft, size=512)
        fake_image = self.__generator.forward(line, draft)

        draft_zero = self.__draft_model.forward(line_draft, zero_hint)
        draft_zero = nn.functional.interpolate(draft_zero, size=512)
        fake_zero_hint_image = self.__generator.forward(line, draft_zero)

        loss = self.__l1_loss(fake_image, target)

        self.tb.add_scalar('TESTING/Generator.loss',
                           loss.item(), self.itr)

        r, g, b, a = torch.chunk(hint, 4, 1)
        hint = torch.cat([r, g, b], 1)
        hint = nn.functional.interpolate(hint, size=512)

        batch_size = target.size(0)
        log_image = [make_grid(target, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(fake_image, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(fake_zero_hint_image, batch_size, 0,
                               range=(-1, 1), normalize=True),
                     make_grid(draft, batch_size, 0,
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
                'adma': self.__optimizer,
                'scheduler': self.__scheduler}

        torch.save(ckpt, file_name)
