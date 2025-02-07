import torch
import torch.nn as nn
import torch.nn.functional as F

class AEBlock(nn.Module):
    def __init__(self, in_features, out_features, n_layers=2, dropout_p=0, act=F.relu, norm=True):
        super(AEBlock, self).__init__()
        self.keep = nn.ModuleList([nn.Linear(in_features, in_features) for i in range(n_layers-1)])
        self.proj = nn.Linear(in_features, out_features)
        self.act = act
        self.apply_norm = norm
        self.dropout_p = dropout_p
        if self.apply_norm:
            self.keep_norm = nn.ModuleList([nn.BatchNorm1d(in_features) for i in range(n_layers-1)])
            self.proj_norm = nn.BatchNorm1d(out_features)
        self.keep_dropout = nn.ModuleList([nn.Dropout(p=self.dropout_p) for i in range(n_layers-1)])
        self.proj_dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        for i, layer in enumerate(self.keep):
            x = self.act(self.keep_norm[i](layer(x))) + x if self.apply_norm else self.act(layer(x)) + x
            x = self.keep_dropout[i](x)
        x = self.act(self.proj_norm(self.proj(x))) if self.apply_norm else self.act(self.proj(x))
        x = self.proj_dropout(x)
        return x


class CVAE(nn.Module):
    def __init__(self, n_blocks, in_features, ratio, n_layers, latent_d, dropout_p=0, act=F.relu, norm=True, c_features=0, end_relu=False, clamp_max=None, clamp_min=None):
        super(CVAE, self).__init__()
        self.dropout_p = dropout_p
        self.enc_widths = [in_features]
        self.latent_d = latent_d
        for i in range(n_blocks):
            self.enc_widths.append(self.enc_widths[-1] // ratio)
        self.enc_widths.append(self.latent_d)
        self.encoder = nn.ModuleList([AEBlock(self.enc_widths[i], self.enc_widths[i+1], n_layers, dropout_p, act, norm) for i in range(len(self.enc_widths) - 1)])
        self.c_proj = nn.Sequential(
            nn.Linear(c_features, self.latent_d),
            nn.GELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.latent_d, self.latent_d)
        )
        self.dec_widths = list(reversed(self.enc_widths))
        self.dec_widths[0] = self.dec_widths[0] * 2
        self.decoder = nn.ModuleList([AEBlock(self.dec_widths[i], self.dec_widths[i+1], n_layers, dropout_p, act, norm) for i in range(len(self.enc_widths) - 1)])
        self.mean_layer = nn.Linear(self.enc_widths[-1]*2, self.enc_widths[-1])
        self.logvar_layer = nn.Linear(self.enc_widths[-1]*2, self.enc_widths[-1])
        self.out_layer = nn.Linear(self.dec_widths[-1], self.dec_widths[-1])
        self.end_relu = end_relu
        self.c_features = c_features
        self.clamp_max = clamp_max
        self.clamp_min = clamp_min

    def reparameterization(self, mean, logvar):
        if self.training:
            epsilon = torch.randn_like(logvar, device=logvar.device)
            z = mean + torch.exp(0.5 * logvar) * epsilon
        else:
            z = mean
        return z

    def forward(self, x, c):
        for layer in self.encoder:
            x = layer(x)
        if self.c_features > 0:
            c = self.c_proj(c)
            x = torch.cat((x,c),dim=1)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterization(mu, logvar)
        y = torch.cat((z, c), dim=1) if self.c_features > 0 else z
        for layer in self.decoder:
            y = layer(y)
        y = self.out_layer(y)
        y = F.relu(y) if self.end_relu else y
        return y, mu, logvar

    def sample(self, c, specified_latent=None):
        c = self.c_proj(c) if self.c_features > 0 else c
        z = torch.randn((c.shape[0],self.enc_widths[-1]), device=c.device) if specified_latent is None else specified_latent
        y = torch.cat((z, c), dim=1) if self.c_features > 0 else z
        for layer in self.decoder:
            y = layer(y)
        y = self.out_layer(y)
        y = F.relu(y) if self.end_relu else y
        if (self.clamp_min is not None) or (self.clamp_max is not None):
            y = torch.clamp(y, min=self.clamp_min, max=self.clamp_max)
        return y

def construct_model(settings):
    model = CVAE(settings.n_blocks, settings.in_features, settings.ratio, settings.n_layers, settings.latent_d, \
        dropout_p=settings.dropout_p, norm=settings.norm, c_features=settings.n_traits, end_relu=settings.end_relu)
    return model

def construct_optimizer(settings, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.base_lr, weight_decay=settings.weight_decay)
    return optimizer








