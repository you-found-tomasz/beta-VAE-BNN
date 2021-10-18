import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
import torchvision
from torch import nn
import vae_decoder_lit_models
import vae_encoder_lit_models
import utils


class VaeModel(pl.LightningModule):

    def __init__(self, config, args):
        super().__init__()

        self.save_hyperparameters()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.latent_space = config["latent_space"]
        self.encoder_type = args.network_enc
        self.decoder_type = args.network_dec
        self.number_classes = args.number_classes
        self.decoder = vae_decoder_lit_models.load_model(config, self.decoder_type, self.number_classes, args.data,
                                                         args.seq_length, self.y_decoder, self.single_to_decoder)
        self.encoder = vae_encoder_lit_models.load_model(config, self.encoder_type, self.number_classes, args.data,
                                                         args.seq_length, self.y_encoder)
        self.encoder_single = vae_encoder_lit_models.Deep_single(config, self.number_classes, args.data, args.seq_length, self.y_encoder)
        self.cluster = args.cluster
        self.data = args.data
        self.number_params_dec = sum(p.numel() for p in self.decoder.parameters())
        self.number_params_enc = sum(p.numel() for p in self.encoder.parameters())
        self.number_params = self.number_params_dec + self.number_params_enc
        self.gpu_active = args.gpu
        self.seq_length = args.seq_length
        self.raytune = args.raytune
        self.images = args.images
        self.counter = 0
        self.beta = args.beta
        self.data_location = args.data_location
        self.alpha = args.alpha
        self.tanh = torch.nn.Tanh()
        self.logging_string = args.logging_string
        self.tanh_bool = False
        self.decoder_only = "True"
        self.MSELoss = torch.nn.MSELoss()
        self.shift = 0
        self.blur = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_short = x[:, 0, :, self.shift:self.seq_length + self.shift]
        x1_single_mu, x1_single_var  = self.encoder_single(x_short)
        spike_count, _, _, _, _ = self.spike_counter(x_short)
        x_short = x_short.reshape(x.shape[0], -1)
        y_decoded = utils.y_decoder(y)

        if self.encoder_type != "Resnet":
            x_encoded_mu, x_encoded_var = self.encoder(x_short, y_decoded)
        else:
            x_image = x[:, :, :, :4096]
            x_image = torch.kron(x_image, torch.ones((2, 2), device=self.device))
            x_image = torch.reshape(x_image, (x.shape[0], -1, 256, 256))
            x_encoded_mu, x_encoded_var = self.encoder(x_image, y_decoded)

        x_encoded_mu = torch.cat([x_encoded_mu, x1_single_mu], dim=1)
        x_encoded_var = torch.cat([x_encoded_var, x1_single_var], dim=1)

        # sample z from q
        std = torch.exp(x_encoded_var / 2)
        q = torch.distributions.Normal(x_encoded_mu, std)
        z = q.rsample()

        # decoded
        if self.single_to_decoder == "False":
            x_hat = self.decoder(z, y_decoded)
        else:
            x_hat = self.decoder(z[:,-16:], y_decoded)
            x_hat_single = self.decoder_single(z)
            x_hat = (x_hat + x_hat_single)/2

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x_short)

        # spike loss
        x_hat = x_hat.reshape(x.shape[0], 4, -1)
        spike_count_x_hat, _, _, _, _ = self.spike_counter(x_hat)
        spike_loss = (spike_count - spike_count_x_hat).float()
        x_short_reshaped = x_short.reshape(x.shape[0], 4, -1)

        # kl
        kl = self.kl_divergence(z, x_encoded_mu, std)

        # elbo
        elbo = (self.beta * kl - recon_loss + self.alpha * spike_loss)
        # elbo = (self.beta * kl - single_loss + self.alpha * spike_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'spike_loss': spike_loss.mean(),
            'elbo_without_spike': (-recon_loss.mean() + self.beta * kl.mean()),
        })

        return {"loss": elbo}

    def validation_step(self, batch, batch_idx):
        if (self.trainer.max_epochs - 1) == self.current_epoch and self.images == "True" and self.counter < 500:
            x, y = batch

            x_short = x[:, 0, :, self.shift:self.seq_length + self.shift]
            x1_single_mu, x1_single_var = self.encoder_single(x_short)
            spike_count, _, _, _, _ = self.spike_counter(x_short)
            x_short = x_short.reshape(x.shape[0], -1)
            y_decoded = utils.y_decoder(y)

            if self.encoder_type != "Resnet":
                x_encoded_mu, x_encoded_var = self.encoder(x_short, y_decoded)
            else:
                x_image = x[:, :, :, :4096]
                x_image = torch.kron(x_image, torch.ones((2, 2), device=self.device))
                x_image = torch.reshape(x_image, (x.shape[0], -1, 256, 256))
                x_encoded_mu, x_encoded_var = self.encoder(x_image, y_decoded)

            x_encoded_mu = torch.cat([x_encoded_mu, x1_single_mu], dim=1)
            x_encoded_var = torch.cat([x_encoded_var, x1_single_var], dim=1)

            # sample z from q
            std = torch.exp(x_encoded_var / 2)
            q = torch.distributions.Normal(x_encoded_mu, std)
            z = q.rsample()

            # x_hat = self.decoder(z)
            if self.single_to_decoder == "False":
                x_hat = self.decoder(q.mean, y)
            else:
                x_hat = self.decoder(q.mean[:,-16:], y)
                x_hat_single = self.decoder_single(q.mean)
                x_hat = (x_hat + x_hat_single) / 2

            self.counter += 1
            return {"latent_space": q.mean, "y": y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("ptl/elbo_loss", avg_loss)
        self.log("ptl/number_params", self.number_params)

    def validation_epoch_end(self, outputs):
        if outputs != []:
            print("adding embedder")

            zeds = torch.stack([x["latent_space"][0] for x in outputs])
            ys = torch.stack([x["y"][0] for x in outputs])
            self.logger.experiment.add_embedding(zeds, metadata=ys.tolist(), global_step=self.counter)
            self.counter = 0

    def prepare_data(self):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #transforms.RandomAffine(translate=(0.2, 0), degrees=0, fill=255)
        ])

        data_full = torchvision.datasets.ImageFolder(self.data_location,transform=train_transform)

        if self.number_classes == 3:
            data_full_samples_0 = [(single_class[0], 0) for single_class in data_full.samples if
                                   (single_class[1] == 57)]  # 57 is stimulus 38,99 is stimulus 76
            data_full_samples_1 = [(single_class[0], 1) for single_class in data_full.samples if
                                   (single_class[1] == 99)]  # 57 is stimulus 38,99 is stimulus 76
            data_full_samples_2 = [(single_class[0], 2) for single_class in data_full.samples if
                                   (single_class[1] == 28)]  # 57 is stimulus 38,99 is stimulus 76
            data_full.samples = data_full_samples_0 + data_full_samples_1 + data_full_samples_2

            data_full_targets_0 = [0 for single_class in data_full.targets if (single_class == 57)]
            data_full_targets_1 = [1 for single_class in data_full.targets if (single_class == 99)]
            data_full_targets_2 = [2 for single_class in data_full.targets if (single_class == 28)]
            data_full.targets = data_full_targets_0 + data_full_targets_1 + data_full_targets_2

            print("class 0: {}".format(data_full_samples_0[0]))
            print("class 1: {}".format(data_full_samples_1[0]))
            print("class 2: {}".format(data_full_samples_2[0]))

        train_size = int(0.9 * len(data_full))
        test_size = len(data_full) - train_size
        print("full data: {}, train size: {}, val size: {}".format(len(data_full), train_size, test_size))
        print(data_full.samples[0])

        _, self.test_dataset = torch.utils.data.random_split(data_full, [train_size, test_size],
                                                             generator=torch.Generator().manual_seed(
                                                                 42))
        self.train_dataset = data_full

    def train_dataloader(self):
        if self.gpu_active != 0 and self.raytune == "False":
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=4, persistent_workers=True)
        else:
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader

    def val_dataloader(self):
        if self.gpu_active != 0 and self.raytune == "False":
            val_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4, shuffle=False, num_workers=2,
                                                     persistent_workers=True)
        else:
            val_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=4, shuffle=False)
        return val_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
