import argparse
from collections import OrderedDict

from zmq import device
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import utils

def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    """Train the model for one epoch.

    :param model: the model (VAE).
    :param train_loader: data loader of training samples.
    :param optimizer: the optimizer, for example, Adam.
    :param epoch: current epoch.
    :param quiet: whether to show training process.
    :param grad_clip: whether to apply grad clip.
    :return: a tuple of losses.
    """

    # switch to train mode
    model.train()

    if not quiet:
        progress_bar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        # move tensor to gpu
        x = x.to(device)
        # calculate loss
        out = model.loss(x)
        # clear grad
        optimizer.zero_grad()
        # calculate gradients of network parameters
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # update parameters
        optimizer.step()

        desc = f'Epoch {epoch}'
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f', {k} {avg_loss:.4f}'

        if not quiet:
            progress_bar.set_description(desc)
            progress_bar.update(x.shape[0])
    if not quiet:
        progress_bar.close()
    return losses


def eval(model, data_loader, quiet):
    """Evaluate the model.
    """

    # switch to eval mode
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = 'Test '
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f', {k} {total_losses[k]:.4f}'
        if not quiet:
            print(desc)
    return total_losses


def train_epochs(model, train_loader, test_loader, args, quiet=False):
    # hyper-parameters
    epochs, lr, grad_clip = args.epochs, args.lr, args.grad_clip
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    for epoch in range(epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
    return train_losses, test_losses



class FullyConnectedVAE(nn.Module):
    """VAE with only fully connected layers.
    """

    def __init__(self, input_dim, latent_dim, enc_hidden_dims, dec_hidden_dims):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,enc_hidden_dims),
            nn.Linear(enc_hidden_dims,enc_hidden_dims),
            nn.Linear(enc_hidden_dims,2*latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,dec_hidden_dims),
            nn.Linear(dec_hidden_dims,dec_hidden_dims),
            nn.Linear(dec_hidden_dims,input_dim),
            nn.Sigmoid()
        )

    def loss(self, x):
        # TODO
        # perform forward propagation and calculate loss
        batch_size=x.shape[0]
        x=x.view(batch_size,-1)
        mu, log_std = self.encoder(x).chunk(2, dim=1)
        sampled_z = self.reparametrizer(mu,log_std)
        x_hat = self.decoder(sampled_z)
        # print(x,x_hat)
        reconstruction_function = nn.BCELoss()
        reconstruction_function.size_average = False
        recon_loss=reconstruction_function(x_hat,x)
        KLD_element = mu.pow(2).add_(log_std.exp()).mul_(-1).add_(1).add_(log_std)
        kl_loss=torch.sum(KLD_element).mul_(-0.5)
        return OrderedDict(loss=recon_loss + kl_loss, recon_loss=recon_loss, kl_loss=kl_loss)

    def reparametrizer(self,mu,log_std):
        std=log_std
    
    def sample(self, n, noise=True):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(device)
            mu, log_std = self.decoder(z).chunk(2, dim=1)
            if noise:
                z = torch.randn_like(mu) * log_std.exp() + mu
            else:
                z = mu
        return z.cpu().numpy()


def train_vae_and_sample(train_data, test_data, args: argparse.Namespace):
    """Train our VAE then use it to generate samples.

    :param train_data: an (n_train, 2) numpy array of floats.
    :param test_data: an (n_test, 2) numpy array of floats.
    :param args: a dict of hyper-parameters.
    :return: a tuple, including

        - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
            and KL term E[KL(q(z|x) | p(z))] evaluated every mini-batch.
        - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruction loss E[-p(x|z)],
            and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch.
        - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z).
        - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z).
    """

    model = FullyConnectedVAE(2, 2, 128, 128).to(device)
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader, args, quiet=True)
    train_losses = np.stack((train_losses['loss'], train_losses['recon_loss'], train_losses['kl_loss']), axis=1)
    test_losses = np.stack((test_losses['loss'], test_losses['recon_loss'], test_losses['kl_loss']), axis=1)

    samples_noise = model.sample(1000, noise=True)
    samples_no_noise = model.sample(1000, noise=False)

    return train_losses, test_losses, samples_noise, samples_no_noise


def main(args: argparse.Namespace):
    # part a
    utils.visualize_data('a', 1)
    utils.save_results('a', 1, train_vae_and_sample, args)

    utils.visualize_data('a', 2)
    utils.save_results('a', 2, train_vae_and_sample, args)

    # part b
    utils.visualize_data('b', 1)
    utils.save_results('b', 1, train_vae_and_sample, args)

    utils.visualize_data('b', 2)
    utils.save_results('b', 2, train_vae_and_sample, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto-encoder')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--grad-clip', default=None, type=float,
                        help='a scalar specifies grad clip (Default: None)')
    parser.add_argument('--device',default='cpu',type=str,help='use which device')
    args = parser.parse_args()
    device= args.device
    main(args)
