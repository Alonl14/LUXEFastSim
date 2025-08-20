"""
trainer.py – critic terminology, W-distance/GP logging, safe break, input jitter,
             LR warmup, early-stop on flat slopes (unchanged behavior but cleaner names).
"""

import warnings
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import utils

warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*You can silence this warning by not passing in num_features.*")


# ------------------------- helpers ------------------------- #

def compute_gradient_penalty(critic: nn.Module, real, fake, lam, device):
    b = real.size(0)
    alpha = torch.rand(b, 1, device=device).expand_as(real)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    d_inter = critic(inter)
    grad = torch.autograd.grad(d_inter, inter, torch.ones_like(d_inter, device=device),
                               create_graph=True, retain_graph=True)[0]
    grad_norm = grad.view(b, -1).norm(2, dim=1)
    return lam * ((grad_norm - 1.0) ** 2).mean()


def compute_r1_gradient_penalty(critic: nn.Module, real, lam):
    b = real.size(0)
    real.requires_grad_(True)
    loss_real = critic(real)
    grad = torch.autograd.grad(loss_real, real, create_graph=True,
                               grad_outputs=torch.ones_like(loss_real))[0]
    r1 = (grad.view(b, -1).norm(2, dim=1) ** 2).mean()
    return 0.5 * lam * r1


def param_grad_norm(net: nn.Module) -> float:
    s = 0.0
    for p in net.parameters():
        if p.grad is not None:
            s += p.grad.detach().norm(2).item() ** 2
    return s ** 0.5


# --------------------------- Trainer --------------------------- #

class Trainer:
    WARMUP_STEPS = 1_000
    SLOPE_EPS = 2e-5
    SLOPE_WINDOW = 12  # epochs

    def __init__(self, cfg):
        # networks & data
        self.genNet = cfg['genNet']
        self.critic = cfg['criticNet']
        self.dl_train = cfg['dataloader']
        self.dl_val = cfg['valDataloader']
        self.dataGroup = cfg['dataGroup']

        # hparams
        self.noiseDim = int(cfg['noiseDim'])
        self.numEpochs = int(cfg['numEpochs'])
        self.device = cfg['device']
        self.nCrit = int(cfg['nCrit'])
        self.Lambda = float(cfg['Lambda'])

        # checkpoints present?
        self.trained = cfg.get('genStateDict') is not None and cfg.get('criticStateDict') is not None

        self.optG = cfg['genOptimizer']
        self.optD = cfg['criticOptimizer']
        self.outputDir = cfg['outputDir']
        self.GMaxSteps = cfg.get('GMaxSteps', None)
        self.gradMetric = cfg.get('gradMetric', 'norm')

        # model checkpoint
        self.best_ValD = float('inf')

        # logs
        self.G_loss_log = []
        self.Critic_wdist_log = []       # formerly D_wdist_log (positive)
        self.Val_G_loss_log = []
        self.Val_Critic_wdist_log = []   # formerly Val_D_log (positive)
        self.KL_log = []
        self.GP_log = []
        self.GP_mean_log = []
        self.gradG_log = []
        self.gradCritic_log = []         # formerly gradD_log

        # KL buffers
        self.real_buf, self.fake_buf = [], []
        self.KL_TARGET = 8192

        # LR warm-up
        self.step_G = 0
        self.schedG = LambdaLR(self.optG, lr_lambda=self._warmup)
        self.schedD = LambdaLR(self.optD, lr_lambda=self._warmup)

    # --------------------- utils --------------------- #
    def _warmup(self, step):
        step = min(step, self.WARMUP_STEPS)
        return 0.5 * (1.0 - np.cos(step / self.WARMUP_STEPS * np.pi))

    # --------------------- main ---------------------- #
    def run(self):
        # init weights only if not loaded from state dict
        if not self.trained:
            print("Initializing weights…")
            utils.weights_init(self.genNet)
            utils.weights_init(self.critic)

        self.genNet.to(self.device).train()
        self.critic.to(self.device).train()

        sigma, decay = 1e-2, 0.9998  # instance noise params
        step_break = False

        for epoch in tqdm.tqdm(range(self.numEpochs), desc='epochs'):
            if step_break:
                break
            sum_G = sum_W = gp_acc = 0.0
            n_batches = 0

            for real in self.dl_train:
                if self.GMaxSteps is not None and self.step_G >= self.GMaxSteps:
                    step_break = True
                    break

                real = real.to(self.device) + sigma * torch.randn_like(real, device=self.device)
                bs = real.size(0)

                # ---- critic ----
                w_dist_batch = 0.0
                gp_batch = 0.0
                for _ in range(self.nCrit):
                    self.optD.zero_grad()

                    loss_real = -self.critic(real).mean()
                    noise = torch.randn(bs, self.noiseDim, device=self.device)
                    fake = self.genNet(noise)
                    fake_noisy = fake + sigma * torch.randn_like(fake, device=self.device)
                    loss_fake = self.critic(fake_noisy.detach()).mean()

                    gp = compute_gradient_penalty(self.critic, real, fake_noisy.detach(),
                                                  self.Lambda, self.device) if self.gradMetric == 'norm' else \
                         compute_r1_gradient_penalty(self.critic, real, self.Lambda)
                    loss_D = loss_real + loss_fake + gp
                    loss_D.backward()
                    self.optD.step()

                    w_dist_batch += (loss_real + loss_fake).item()
                    gp_batch += gp.item()

                # ---- generator ----
                self.optG.zero_grad()
                noise = torch.randn(bs, self.noiseDim, device=self.device)
                fake = self.genNet(noise)
                fake_noisy = fake + sigma * torch.randn_like(fake, device=self.device)
                loss_G = -self.critic(fake_noisy).mean()
                loss_G.backward()
                self.optG.step()

                # decay noise std
                sigma *= decay

                # schedulers & counters
                self.step_G += 1
                self.schedG.step()
                self.schedD.step()

                # per-batch logs (less frequent)
                if self.step_G % 500 == 0:
                    self.GP_log.append(gp_batch / self.nCrit)
                    self.gradG_log.append(param_grad_norm(self.genNet))
                    self.gradCritic_log.append(param_grad_norm(self.critic))

                sum_G += loss_G.item()
                sum_W += w_dist_batch / self.nCrit
                gp_acc += gp_batch
                n_batches += 1

                # mini-KL (on target buffer size)
                self.real_buf.append(real.detach().cpu())
                self.fake_buf.append(fake.detach().cpu())
                if sum(t.size(0) for t in self.real_buf) >= self.KL_TARGET:
                    r = torch.cat(self.real_buf)[:self.KL_TARGET]
                    f = torch.cat(self.fake_buf)[:self.KL_TARGET]
                    self.KL_log.append(utils.get_kld(r.to(self.device), f.to(self.device)).item())
                    self.real_buf.clear()
                    self.fake_buf.clear()

            if n_batches == 0:
                break  # no data processed

            self.G_loss_log.append(sum_G / n_batches)
            self.Critic_wdist_log.append(-(sum_W / n_batches))  # make it positive (W estimate)
            self.GP_mean_log.append(gp_acc / max(1, n_batches * self.nCrit))

            vG, vD = self._validate()
            self.Val_G_loss_log.append(vG)
            self.Val_Critic_wdist_log.append(-vD)

            # checkpoint: best validation critic (positive & decreasing)
            if self.best_ValD > -vD > 0:
                self.best_ValD = -vD
                print(f"New best Val_Critic_W: {self.best_ValD:.4f}")
                torch.save(self.genNet.state_dict(),
                           os.path.join(self.outputDir, f"{self.dataGroup}_Gen_model.pt"))
                torch.save(self.critic.state_dict(),
                           os.path.join(self.outputDir, f"{self.dataGroup}_Critic_model.pt"))

            # --- early-stop slopes ---
            win = min(self.SLOPE_WINDOW, len(self.KL_log), len(self.Critic_wdist_log))
            if win >= 3 and epoch > 60:
                def slope(arr):
                    try:
                        return np.polyfit(range(win), arr[-win:], 1)[0]
                    except np.linalg.LinAlgError:
                        return 0.0

                if abs(slope(self.KL_log)) < self.SLOPE_EPS and abs(slope(self.Critic_wdist_log)) < self.SLOPE_EPS:
                    print('Early stop: slopes flat')
                    break

    # --------------------- validate ------------------ #
    def _validate(self):
        self.genNet.eval()
        self.critic.eval()
        sG = sD = n = 0
        with torch.no_grad():
            for real in self.dl_val:
                real = real.to(self.device)
                bs = real.size(0)
                loss_real = -self.critic(real).mean()
                fake = self.genNet(torch.randn(bs, self.noiseDim, device=self.device))
                loss_fake = self.critic(fake).mean()
                sD += (loss_real + loss_fake).item()
                sG += (-self.critic(fake).mean()).item()
                n += 1
        self.genNet.train()
        self.critic.train()
        return sG / n, sD / n
