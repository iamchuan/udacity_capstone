import os
import time

import torch
from torchvision import utils as vutils
from .utils.functions import *


class GANTrainer(object):

    def __init__(self, generator, discriminator, optimizer, loss,
                 dtype=torch.float, device='cuda', multi_gpu=True,
                 num_epochs=10, batch_size=64, noize_dim=100, sample_size=100,
                 output_path='./output',
                 lr=0.001, beta1=0.5):
        # arguments
        self.batch_size = batch_size
        self.noize_dim = noize_dim
        self.num_epochs = num_epochs
        self.dtype = dtype
        self.output_path = output_path
        self.lr = lr
        self.beta1 = beta1

        if isinstance(device, torch.device):
            self.device = device
        elif torch.cuda.is_available() and device.find('cuda') == 0:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        self.multi_gpu = multi_gpu

        # networks
        self.netG = generator
        self.netD = discriminator
        # optimizers
        self.optimizer = optimizer
        # loss
        self.loss = loss
        # fixed noise
        self.fixed_noise = torch.randn(sample_size, self.noize_dim,
                                       dtype=self.dtype, device=self.device,
                                       requires_grad=False)

    def save_samples(self, filename, noise=None, **kwargs):
        if noise is None:
            noise = self.fixed_noise
        with torch.no_grad():
            vutils.save_image(self.netG(noise).detach(), filename,
                              normalize=True, range=(0, 1), **kwargs)

    def train(self, dataloader, nrow=8):
        dtype = self.dtype
        device = self.device

        # networks
        netG = self.netG
        netD = self.netD
        # DataParallel on multiple GPUs
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print("GPUs Available: ", torch.cuda.device_count())
            netG = nn.DataParallel(netG)
            netD = nn.DataParallel(netD)

        netG = netG.to(device)
        netD = netD.to(device)

        # parameter initilization
        netG.apply(weights_init)
        netD.apply(weights_init)
        # optmizers
        optG = self.optimizer(netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optD = self.optimizer(netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # loss
        loss = self.loss
        # fake samples
        fake_sample_path = os.path.join(self.output_path, 'fake_samples_epoch_{:03d}.png')

        self.save_samples(filename=fake_sample_path.format(0), nrow=nrow)

        # losses
        self.d_losses = list()
        self.g_losses = list()

        start = time.time()
        for epoch in range(self.num_epochs):
            lap_start = time.time()
            for i, (imgs, y) in enumerate(dataloader):
                # batch size
                _batch_size = imgs.size(0)
                # Input images
                # real images
                real_imgs = imgs.to(device)
                # fake images generated from gaussian noise
                noise = torch.randn(_batch_size, self.noize_dim,
                                    dtype=self.dtype, device=self.device, requires_grad=False)
                fake_imgs = netG(noise)
                # Adversarial ground truths
                real_label = torch.full((_batch_size,), 1, dtype=dtype, device=device, requires_grad=False)
                fake_label = torch.full((_batch_size,), 0, dtype=dtype, device=device, requires_grad=False)

                #########################
                # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                #########################

                optD.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = loss(netD(real_imgs), real_label)
                fake_loss = loss(netD(fake_imgs.detach()), fake_label)
                d_loss = real_loss + fake_loss

                d_loss.backward()
                optD.step()


                #########################
                # Update Generator: maximize log(D(G(z)))
                #########################

                optG.zero_grad()

                # measures generator's ability to fool the discriminator
                g_loss = loss(netD(fake_imgs), real_label)

                g_loss.backward()
                optG.step()

                # losses

                self.d_losses.append(d_loss.item())
                self.g_losses.append(g_loss.item())
                # training logging
                if i % (len(dataloader) // 5) == 0:
                    print('[Epoch {}/{}][Batch {}/{}] Loss_D: {:.2f} Loss_G: {:.2f}'.format(epoch, self.num_epochs,
                                                                                            i, len(dataloader),
                                                                                            d_loss.item(),
                                                                                            g_loss.item()))
            else:
                print('Epoch {} running time: {:.2f}'.format(epoch, time.time() - lap_start))
                self.save_samples(filename=fake_sample_path.format(epoch + 1), nrow=nrow)

        print('Total running time: {:.2f}s'.format(time.time() - start))


class CGANTrainer(GANTrainer):

    def __init__(self, generator, discriminator, optimizer, loss, y_fixed,
                 dtype=torch.float, device='cuda', multi_gpu=True,
                 num_epochs=10, batch_size=64, noize_dim=100, sample_size=100,
                 output_path='./output', lr=0.001, beta1=0.5):
        super(CGANTrainer, self).__init__(generator, discriminator, optimizer, loss,
                                          dtype, device, multi_gpu, num_epochs, batch_size,
                                          noize_dim, sample_size, output_path, lr, beta1)
        self.y_fixed = y_fixed

    def save_samples(self, filename, noise=None, y=None, **kwargs):
        if noise is None:
            noise = self.fixed_noise
        if y is None:
            y = self.y_fixed
        with torch.no_grad():
            vutils.save_image(self.netG(noise, y).detach(), filename,
                              normalize=True, range=(0, 1), **kwargs)

    def train(self, dataloader, nrow=8):
        dtype = self.dtype
        device = self.device

        # networks
        netG = self.netG
        netD = self.netD
        # DataParallel on multiple GPUs
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print("GPUs Available: ", torch.cuda.device_count())
            netG = nn.DataParallel(netG)
            netD = nn.DataParallel(netD)

        netG = netG.to(device)
        netD = netD.to(device)

        # parameter initilization
        netG.apply(weights_init)
        netD.apply(weights_init)
        # optmizers
        optG = self.optimizer(netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optD = self.optimizer(netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        # loss
        loss = self.loss
        # fake samples
        fake_sample_path = os.path.join(self.output_path, 'fake_samples_epoch_{:03d}.png')

        self.save_samples(filename=fake_sample_path.format(0), nrow=nrow)

        # losses
        self.d_losses = list()
        self.g_losses = list()

        start = time.time()
        for epoch in range(self.num_epochs):
            lap_start = time.time()
            for i, (imgs, y) in enumerate(dataloader):
                # batch size
                _batch_size = imgs.size(0)
                # Input images
                # real images
                real_imgs = imgs.to(device)
                y = y.to(device)
                # fake images generated from gaussian noise
                noise = torch.randn(_batch_size, self.noize_dim,
                                    dtype=self.dtype, device=self.device, requires_grad=False)
                y_noise = torch.randint(10, (_batch_size, ), device=device)
                fake_imgs = netG(noise, y_noise)
                # Adversarial ground truths
                real_label = torch.full((_batch_size,), 1, dtype=dtype, device=device, requires_grad=False)
                fake_label = torch.full((_batch_size,), 0, dtype=dtype, device=device, requires_grad=False)

                #########################
                # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                #########################

                optD.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = loss(netD(real_imgs, y), real_label)
                fake_loss = loss(netD(fake_imgs.detach(), y_noise), fake_label)
                d_loss = real_loss + fake_loss

                d_loss.backward()
                optD.step()


                #########################
                # Update Generator: maximize log(D(G(z)))
                #########################

                optG.zero_grad()

                # measures generator's ability to fool the discriminator
                g_loss = loss(netD(fake_imgs, y_noise), real_label)

                g_loss.backward()
                optG.step()

                # losses

                self.d_losses.append(d_loss.item())
                self.g_losses.append(g_loss.item())
                # training logging
                if i % (len(dataloader) // 5) == 0:
                    print('[Epoch {}/{}][Batch {}/{}] Loss_D: {:.2f} Loss_G: {:.2f}'.format(epoch, self.num_epochs,
                                                                                            i, len(dataloader),
                                                                                            d_loss.item(),
                                                                                            g_loss.item()))
            else:
                print('Epoch {} running time: {:.2f}'.format(epoch, time.time() - lap_start))
                self.save_samples(filename=fake_sample_path.format(epoch + 1), nrow=nrow)

        print('Total running time: {:.2f}s'.format(time.time() - start))

