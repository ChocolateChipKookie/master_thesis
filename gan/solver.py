import torch
from torch.utils.data import DataLoader

from util import module, util

class GANsolver(module.Solver):

    def __init__(self, network, config):
        super(GANsolver, self).__init__(network, config)
        # Create discriminator
        self.net_G = self.network
        self.opt_G = self.optimizer
        self.net_D = util.factory(self.derived_config["discriminator"])
        self.net_D = self.net_D.to(self.device)
        optimizer_data = self.solver_config['optimizer']
        optimizer_class = util.import_attr(optimizer_data['class'])
        self.opt_D = optimizer_class(self.net_D.parameters(), **optimizer_data['args'])

        self.valid_label = torch.tensor(self.derived_config['valid_label']).to(self.device)
        self.invalid_label = torch.tensor(self.derived_config['invalid_label']).to(self.device)
        self.gan_loss = util.factory(self.derived_config["gan_loss"])
        self.cond_loss = util.factory(self.derived_config["cond_loss"])
        self.cond_lambda = self.derived_config["cond_lambda"]

        # If restore is enabled, restore the discriminator state dict
        if self.restore:
            disc_state_dict = torch.load(self.state['discriminator_state_dict'])
            self.net_D.load_state_dict(disc_state_dict)
            self.net_D.eval()

        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None
        self.loss_G_fake = None
        self.loss_G_cond = None
        self.loss_G = None

    def criterion(self, pred_D, target):
        label = self.valid_label if target else self.invalid_label
        label = label.expand_as(pred_D)
        return self.gan_loss(pred_D, label)

    def calculate_loss(self, batch):
        return self.loss_D, self.loss_D_real, self.loss_D_fake, self.loss_G, self.loss_G_cond, self.loss_G_fake

    def eval_D(self, l, ab, fake):
        # Pass fake
        input_D = torch.cat((l, fake), 1)
        predicted_fake = self.net_D(input_D)
        self.loss_D_fake = self.criterion(predicted_fake, False)
        # Pass true
        input_D = torch.cat((l, ab), 1)
        predicted_real = self.net_D(input_D)
        self.loss_D_real = self.criterion(predicted_real, True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2

    def eval_G(self, l, ab, fake):
        # Fake the discriminator
        input_D = torch.cat((l, fake), 1)
        predicted = self.net_D(input_D)
        # Options for minimizing or maximizing:
        # self.loss_G_fake = self.criterion(predicted, True)
        self.loss_G_fake = -self.criterion(predicted, False)
        # L1 loss
        self.loss_G_cond = self.cond_loss(fake, ab) * self.cond_lambda
        self.loss_G = self.loss_G_fake + self.loss_G_cond

    def set_requires_grad(self, net, requires_grad):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def train(self):
        self.network.train()
        while self.iteration <= self.iterations:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=self.loaders
            )

            for i, batch in enumerate(data_loader, self.iteration):
                if self.iteration > self.iterations:
                    break
                self.iteration += 1
                # Reset the gradients
                self.optimizer.zero_grad()
                # Fetch images
                batch, _ = batch

                # Fetch input for the network
                l = batch[:, :1, :, :].to(self.device)
                l_norm = self.net_G.normalize_l(l)
                ab = batch[:, 1:, :, :].to(self.device)
                ab_norm = self.net_G.normalize_ab(ab)

                # Create fake batch
                fake = self.net_G(l_norm, True)

                # Update discriminator
                self.set_requires_grad(self.net_D, True)
                self.opt_D.zero_grad()
                self.eval_D(l_norm, ab_norm, fake)
                self.loss_D.backward(retain_graph=True)
                # Step discriminator
                self.opt_D.step()
                self.set_requires_grad(self.net_D, False)

                # Update generator
                self.opt_G.zero_grad()
                self.eval_G(l_norm, ab_norm, fake)
                self.loss_G.backward()
                # Step generator
                self.opt_G.step()

                # Fetch losses
                loss = [x.item() for x in self.calculate_loss(None)]
                for logger in self.listeners:
                    logger(i, loss)

                # Clean memory
                torch.cuda.empty_cache()