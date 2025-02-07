import torch
import torch.nn.functional as F
from utils import *
from eval import *
import scipy.stats as stats

def vae_loss(x, y, mu, logvar, settings):
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    mse_loss = F.mse_loss(y, x)
    loss = mse_loss + kl_loss * settings.kl_w
    return loss, kl_loss ,mse_loss

def train_step(settings, model, optimizer, x, traits, writer, device):
    x = x.float().to(device)
    traits = traits.float().to(device)
    y, mu, logvar = model(x, traits)
    loss, kl_loss, mse_loss = vae_loss(x, y, mu, logvar, settings)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return y, loss, kl_loss, mse_loss

def train_epoch(settings, model, optimizer, loader, connectivity_preprocess, writer, epoch, device):
    model.train()
    loss, kl_loss, mse_loss = 0, 0, 0
    for i, data in enumerate(loader):
        x_raw = data["connectome"].numpy() # have been log-scaled but not been preprocessed (standardized, and pca if settings.apply_pca is true) yet
        x, x_demean = preprocess_connectivity(x_raw, connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
        x = torch.from_numpy(x) # standardize and pca if settings.apply_pca is true and convert to tensor
        traits = data["traits"] # have been preprocessed (mean imputed and standardized)
        
        y, loss_b, kl_loss_b, mse_loss_b = train_step(settings, model, optimizer, x, traits, writer, device)
        
        if i==0: # evaluate corr in generation for the first minibatch
            gen = model.sample(traits.float().to(device))
            gen_raw, gen_demean = inverse_connectivity_preprocess(gen.detach().cpu().numpy(), connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
            raw_corr = crosscorr(torch.from_numpy(x_raw).float().to(device), torch.from_numpy(gen_raw).float().to(device))
            demean_corr = crosscorr(torch.from_numpy(x_demean).float().to(device), torch.from_numpy(gen_demean).float().to(device))
            standardized_corr = crosscorr(x.float().to(device), gen.float().to(device))
            offdiag_indx = torch.ones_like(raw_corr, dtype=bool)
            raw_diag = torch.diag(raw_corr).detach().cpu().numpy()
            raw_offdiag = raw_corr[offdiag_indx].detach().cpu().numpy()
            demean_diag = torch.diag(demean_corr).detach().cpu().numpy()
            demean_offdiag = demean_corr[offdiag_indx].detach().cpu().numpy()
            standardized_diag = torch.diag(standardized_corr).detach().cpu().numpy()
            standardized_offdiag = standardized_corr[offdiag_indx].detach().cpu().numpy()
            raw_t, raw_p = stats.ttest_ind(raw_diag, raw_offdiag, alternative="greater")
            demean_t, demean_p = stats.ttest_ind(demean_diag, demean_offdiag, alternative="greater")
            standardized_t, standardized_p = stats.ttest_ind(standardized_diag, standardized_offdiag, alternative="greater")

            writer.add_scalar("raw_corr/diag_train", raw_diag.mean(), epoch)
            writer.add_scalar("raw_corr/offdiag_train", raw_offdiag.mean(), epoch)
            writer.add_scalar("raw_corr/p_train", raw_p, epoch)
            writer.add_scalar("demean_corr/diag_train", demean_diag.mean(), epoch)
            writer.add_scalar("demean_corr/offdiag_train", demean_offdiag.mean(), epoch)
            writer.add_scalar("demean_corr/p_train", demean_p, epoch)
            writer.add_scalar("standardized_corr/diag_train", standardized_diag.mean(), epoch)
            writer.add_scalar("standardized_corr/offdiag_train", standardized_offdiag.mean(), epoch)
            writer.add_scalar("standardized_corr/p_train", standardized_p, epoch)

        loss += loss_b
        kl_loss += kl_loss_b
        mse_loss += mse_loss_b

    loss /= i
    kl_loss /= i
    mse_loss /= i

    writer.add_scalar("loss/train", loss.item(), epoch)
    writer.add_scalar("loss_kl/train", kl_loss.item(), epoch)
    writer.add_scalar("loss_mse/train", mse_loss.item(), epoch)
    return loss

def test_model(settings, model, optimizer, loader, connectivity_preprocess, writer, epoch, device):
    model.eval()
    loss, kl_loss, mse_loss = 0, 0, 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_raw = data["connectome"].numpy() # have been log-scaled but not been preprocessed (standardized, and pca if settings.apply_pca is true) yet
            x, x_demean = preprocess_connectivity(x_raw, connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
            x = torch.from_numpy(x) # standardize and pca if settings.apply_pca is true and convert to tensor
            traits = data["traits"] # have been preprocessed (mean imputed and standardized)
            
            x = x.float().to(device)
            traits = traits.float().to(device)
            y, mu, logvar = model(x, traits)
            loss_b, kl_loss_b, mse_loss_b = vae_loss(x, y, mu, logvar, settings)
            
            if i==0: # evaluate corr in generation for the first minibatch
                y_raw, _ = inverse_connectivity_preprocess(y.detach().cpu().numpy(), connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
                gen = model.sample(traits.float().to(device))
                gen_raw, gen_demean = inverse_connectivity_preprocess(gen.detach().cpu().numpy(), connectivity_preprocess, settings.use_raw, settings.apply_pca, settings.n_components)
                raw_corr = crosscorr(torch.from_numpy(x_raw).float().to(device), torch.from_numpy(gen_raw).float().to(device))
                demean_corr = crosscorr(torch.from_numpy(x_demean).float().to(device), torch.from_numpy(gen_demean).float().to(device))
                standardized_corr = crosscorr(x.float().to(device), gen.float().to(device))
                offdiag_indx = torch.ones_like(raw_corr, dtype=bool)
                raw_diag = torch.diag(raw_corr).detach().cpu().numpy()
                raw_offdiag = raw_corr[offdiag_indx].detach().cpu().numpy()
                demean_diag = torch.diag(demean_corr).detach().cpu().numpy()
                demean_offdiag = demean_corr[offdiag_indx].detach().cpu().numpy()
                standardized_diag = torch.diag(standardized_corr).detach().cpu().numpy()
                standardized_offdiag = standardized_corr[offdiag_indx].detach().cpu().numpy()
                raw_t, raw_p = stats.ttest_ind(raw_diag, raw_offdiag, alternative="greater")
                demean_t, demean_p = stats.ttest_ind(demean_diag, demean_offdiag, alternative="greater")
                standardized_t, standardized_p = stats.ttest_ind(standardized_diag, standardized_offdiag, alternative="greater")

                examples = (x_raw[0,:], y_raw[0,:], gen_raw[0,:])

                writer.add_scalar("raw_corr/diag_test", raw_diag.mean(), epoch)
                writer.add_scalar("raw_corr/offdiag_test", raw_offdiag.mean(), epoch)
                writer.add_scalar("raw_corr/diff", raw_diag.mean() - raw_offdiag.mean(), epoch)
                writer.add_scalar("raw_corr/p_test", raw_p, epoch)
                writer.add_scalar("demean_corr/diag_test", demean_diag.mean(), epoch)
                writer.add_scalar("demean_corr/offdiag_test", demean_offdiag.mean(), epoch)
                writer.add_scalar("demean_corr/diff", demean_diag.mean() - demean_offdiag.mean(), epoch)
                writer.add_scalar("demean_corr/p_test", demean_p, epoch)
                writer.add_scalar("standardized_corr/diag_test", standardized_diag.mean(), epoch)
                writer.add_scalar("standardized_corr/offdiag_test", standardized_offdiag.mean(), epoch)
                writer.add_scalar("standardized_corr/p_test", standardized_p, epoch)

            loss += loss_b
            kl_loss += kl_loss_b
            mse_loss += mse_loss_b

        loss /= i
        kl_loss /= i
        mse_loss /= i

    writer.add_scalar("loss/test", loss.item(), epoch)
    writer.add_scalar("loss_kl/test", kl_loss.item(), epoch)
    writer.add_scalar("loss_mse/test", mse_loss.item(), epoch)
    return loss, examples