import pandas as pd 
import csv
import numpy as np
import sklearn
import gzip
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scanpy as sc

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

mk_genes = ["PTPRC", "PAX6", "PDGRFA", "NEUROD2", "GAD1", "AQP4"]
major_cell_types = ["Microglia", "NPCs", "OPCs", "Excitatory_neurons", "Interneurons", "Astrocytes"]
gene2celltype = dict(zip(mk_genes, major_cell_types))


class VAE(nn.Module):
    def __init__(self, input_d):
        super(VAE, self).__init__()
        self.input_d = input_d 
        self.fc1 = nn.Linear(self.input_d, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.input_d)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x):
        h3 = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_d))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_func(x, x_rec, mu, logvar, input_d):
    BCE = F.binary_cross_entropy(x_rec, x.view(-1, input_d), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class VAE(nn.Module):
    def __init__(self, image_size=4490 , h_dim=200, z_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(image_size, h_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(h_dim, z_dim*2)
                )

        self.decoder = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, image_size),
                nn.Sigmoid()
                )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD



def train(data_loader, vae, optimizer, batch_size=128, epochs=100):
    for epoch in range(epochs):
        for idx, cells in enumerate(data_loader):
            cells = to_var(cells)
            recon_cells, mu, logvar = vae(cells)
            loss, bce, kld = loss_fn(recon_cells, cells, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if idx%10000 == 0:
            #    print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.detach()/batch_size))
            #    print(loss.data.detach(), bce.data.detach(), kld.data.detach())
            #    print(cells.shape, recon_cells.shape)
        if epoch%50 == 0:
            print("Epoch[{}/{}] Loss: {:.3f}".format(epoch+1, epochs, loss.data.detach()/batch_size))

