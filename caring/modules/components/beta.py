"""model.py"""

import torch
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from .keypoint import SpatialSoftmax
import ipdb as pdb

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_CNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super(BetaVAE_CNN, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        logvar = logvar - 2
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaVAE_Physics(nn.Module):
    """Visual Encoder/Decoder for Ball dataset."""
    def __init__(self, z_dim=10, nc=3, nf=16, norm_layer='Batch', hidden_dim=512):
        super(BetaVAE_Physics, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        k = 5
        self.k = k
        height = 64
        width = 64
        lim=[-5., 5., -5., 5.]
        self.height = height
        self.width = width
        self.lim = lim
        x = np.linspace(lim[0], lim[1], width // 4)
        y = np.linspace(lim[2], lim[3], height // 4)
        z = np.linspace(-1., 1., k)
        self.register_buffer('x', torch.FloatTensor(x))
        self.register_buffer('y', torch.FloatTensor(y))
        self.register_buffer('z', torch.FloatTensor(z))

        self.integrater = SpatialSoftmax(height=height//4, width=width//4, channel=k, lim=lim)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
        )

        self.fc = nn.Sequential(View((-1, k * 16 * 16)),
                                nn.Linear(k * 16 * 16, z_dim)
                                )

        self.decoder = nn.Sequential(            
            nn.ConvTranspose2d(self.k, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3))

        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def _encode(self, x):
        heatmap = self.encoder(x)
        batch_size = heatmap.shape[0]
        mu = self.integrater(heatmap)
        mu = mu.view(batch_size, -1)
        logvar = self.fc(heatmap)
        return torch.cat((mu,logvar), dim=-1)

    def _decode(self, z):
        kpts = z.view(-1, self.k, 2)
        hmap = self.keypoint_to_heatmap(kpts)
        return self.decoder(hmap)

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, decoder_side_truez=0, decoder_side_hatz=0,decoder_side_truex=0,encoder_side_truez=0, encoder_side_truex=0, hidden_dim=128):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.decoder_side_truez = decoder_side_truez
        self.decoder_side_hatz = decoder_side_hatz
        self.decoder_side_truex = decoder_side_truex
        self.encoder_side_truez = encoder_side_truez
        self.encoder_side_truex = encoder_side_truex
        
        dec_sidevars = [self.decoder_side_truez, self.decoder_side_hatz, self.decoder_side_truex]
        enc_didevars = [self.encoder_side_truez, self.encoder_side_truex]

        # get the non-zero value for decoder_side
        dec_sidelen = next((var for var in dec_sidevars if var != 0), 0)

        # get the non-zero value for encoder_side
        enc_sidelen = next((var for var in enc_didevars if var != 0), 0)


        self.input_dim_sid_enc = input_dim + enc_sidelen * input_dim

        self.input_dim_sid_dec = z_dim + dec_sidelen * z_dim

        self.encoder = nn.Sequential(
                                       nn.Linear(self.input_dim_sid_enc, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(  nn.LeakyReLU(0.2),
                                       nn.Linear(self.input_dim_sid_dec, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )


        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, y, return_z=True):
        batch_size = x.shape[0]
        length = x.shape[1]
       
        if self.encoder_side_truez != 0:
            x_tmp = x.clone()
            x_fuse = [] 
            x_fuse.append(x_tmp)
            for i in range(self.encoder_side_truez):
                y_tmp = y[:,self.encoder_side_truez - (i+1):-(i+1)]
                y_pad = F.pad(y_tmp, (0, 0, self.encoder_side_truez, 0))
                x_fuse.append(y_pad)
                
            x_side = torch.cat(x_fuse, dim=2)
            x_flat = x_side.view(-1, self.input_dim_sid_enc)
            distributions = self._encode(x_flat)
        elif self.encoder_side_truex != 0:
            x_tmp = x.clone()
            x_fuse = [] 
            x_fuse.append(x_tmp)
            for i in range(self.encoder_side_truex):
                x_tmp = x[:,self.encoder_side_truex - (i+1):-(i+1)]
                x_pad = F.pad(x_tmp, (0, 0, self.encoder_side_truex, 0))
                x_fuse.append(x_pad)

            x_side = torch.cat(x_fuse, dim=2)
            x_flat = x_side.view(-1, self.input_dim_sid_enc)
            distributions = self._encode(x_flat)
        else:
            x_flat = x.view(-1, self.input_dim)
            # y_flat = y.view(-1, self.z_dim)
            distributions = self._encode(x_flat)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)

        if self.decoder_side_truez != 0:
            z_tmp = z.clone().reshape(batch_size,length, self.z_dim)
            z_fuse = [] 
            z_fuse.append(z_tmp)
            for i in range(self.decoder_side_truez):
                y_tmp = y[:,self.decoder_side_truez - (i+1):-(i+1)]
                y_pad = F.pad(y_tmp, (0, 0, self.decoder_side_truez, 0))
                z_fuse.append(y_pad)
                
            z_side = torch.cat(z_fuse, dim=2)
            z_flat = z_side.view(-1, self.input_dim_sid_dec)
            x_recon = self._decode(z_flat)
        elif self.decoder_side_hatz != 0:
            zs = [z]

            for i in range(self.decoder_side_hatz):
                z_tmp = z.detach().reshape(batch_size,length, self.z_dim) # BS, LEN, DIM
                z_tmp = z_tmp[:, i+1:, :]
                z_tmp = torch.cat((z_tmp, torch.zeros((batch_size, i+1, self.z_dim)).cuda()), 1)
                z_tmp = z_tmp.reshape(-1, self.z_dim)
                zs.append(z_tmp)

            z_flat = torch.cat(zs, 1)
            x_recon = self._decode(z_flat)

        elif self.decoder_side_truex != 0:
            z_tmp = z.clone().reshape(batch_size,length, self.z_dim)
            z_fuse = [] 
            z_fuse.append(z_tmp)
            for i in range(self.decoder_side_truex):
                x_tmp = x[:,self.decoder_side_truex - (i+1):-(i+1)]
                x_pad = F.pad(x_tmp, (0, 0, self.decoder_side_truex, 0))
                z_fuse.append(x_pad)
            
            z_side = torch.cat(z_fuse, dim=2)
            z_flat = z_side.view(-1, self.input_dim_sid_dec)
            x_recon = self._decode(z_flat)
        else:
            x_recon = self._decode(z)


        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
