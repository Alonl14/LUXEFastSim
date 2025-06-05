"""
trainer.py – tidy‑up (W‑distance, GP logging, safe break, input jitter)
-----------------------------------------------------------------------
Minimal deltas; all list names & public API unchanged.
"""

import torch
import numpy as np
import tqdm
import utils
import warnings
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*You can silence this warning by not passing in num_features.*")

# ------------------------------------------------------------------ #
#  helpers                                                            #
# ------------------------------------------------------------------ #


def compute_gradient_penalty(critic, real, fake, lam, device):
    b = real.size(0)
    alpha = torch.rand(b, 1, device=device).expand_as(real)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    d_inter = critic(inter)
    grad = torch.autograd.grad(d_inter, inter, torch.ones_like(d_inter, device=device),
                               create_graph=True, retain_graph=True)[0]
    grad_norm = grad.view(b, -1).norm(2, dim=1)
    return lam * ((grad_norm - 1) ** 2).mean()


def compute_r1_gradient_penalty(critic, real, lam):
    b = real.size(0)
    real.requires_grad_(True)
    loss_real = critic(real)
    grad = torch.autograd.grad(loss_real, real, create_graph=True,
                               grad_outputs=torch.ones_like(loss_real))[0]
    r1 = (grad.view(b, -1).norm(2, dim=1) ** 2).mean()
    return 0.5 * lam * r1


def param_grad_norm(net):
    s = 0.0
    for p in net.parameters():
        if p.grad is not None:
            s += p.grad.detach().norm(2).item() ** 2
    return s ** 0.5


# ------------------------------------------------------------------ #
class Trainer:
    WARMUP_STEPS = 1_000
    SLOPE_EPS    = 0.002
    SLOPE_WINDOW = 12  # epochs

    def __init__(self, cfg):
        # networks & data
        self.genNet  = cfg['genNet']
        self.discNet = cfg['discNet']
        self.dl_train = cfg['dataloader']
        self.dl_val   = cfg['valDataloader']
        self.dataGroup = cfg['dataGroup']

        # hparams
        self.noiseDim = cfg['noiseDim']
        self.numEpochs = cfg['numEpochs']
        self.device    = cfg['device']
        self.nCrit     = cfg['nCrit']
        self.Lambda    = cfg['Lambda']

        self.optG = cfg['genOptimizer']
        self.optD = cfg['discOptimizer']
        self.outputDir = cfg['outputDir']
        self.GMaxSteps = cfg.get('GMaxSteps', None)
        self.gradMetric = cfg.get('gradMetric', 'norm')

        # model checkpoint
        self.best_ValD = float('inf')

        # logs
        self.G_loss_log   = []
        self.D_wdist_log  = []
        self.Val_G_log    = []
        self.Val_D_log    = []
        self.KL_log       = []
        self.GP_log       = []
        self.GP_mean_log  = []
        self.gradG_log    = []
        self.gradD_log    = []

        # KL buffers
        self.real_buf, self.fake_buf = [], []
        self.KL_TARGET = 8192

        # LR warm‑up
        self.step_G = 0
        self.schedG = LambdaLR(self.optG, lr_lambda=self._warmup)
        self.schedD = LambdaLR(self.optD, lr_lambda=self._warmup)

    # --------------------- utils --------------------- #
    def _warmup(self, step):
        step = min(step, self.WARMUP_STEPS)
        return 0.5 * (1 - np.cos(step / self.WARMUP_STEPS * np.pi))

    # --------------------- main ---------------------- #
    def run(self):
        utils.weights_init(self.genNet)
        utils.weights_init(self.discNet)
        self.genNet.to(self.device).train()
        self.discNet.to(self.device).train()

        sigma, decay = 1e-2, 0.9998  # instance noise params
        step_break = False

        for epoch in tqdm.tqdm(range(self.numEpochs), desc='epochs'):
            if step_break: break
            sum_G = sum_W = gp_acc = 0.0; n_batches = 0

            for real in self.dl_train:
                if self.GMaxSteps is not None:
                    if self.step_G >= self.GMaxSteps:
                        step_break = True; break

                real = real.to(self.device) + sigma * torch.randn_like(real, device=self.device)
                bs = real.size(0)

                # ---- critic ----
                w_dist_batch = 0.0
                gp_batch = 0.0
                for _ in range(self.nCrit):
                    self.optD.zero_grad()

                    loss_real = -self.discNet(real).mean()
                    noise = torch.randn(bs, self.noiseDim, device=self.device)
                    fake = self.genNet(noise)
                    fake_noisy = fake + sigma * torch.randn_like(fake, device=self.device)
                    loss_fake = self.discNet(fake_noisy.detach()).mean()

                    gp = compute_gradient_penalty(self.discNet, real, fake_noisy.detach(),
                                                   self.Lambda, self.device) if self.gradMetric == 'norm' else \
                        compute_r1_gradient_penalty(self.discNet, real, self.Lambda)
                    loss_D = loss_real + loss_fake + gp
                    loss_D.backward(); self.optD.step()

                    w_dist_batch += (loss_real + loss_fake).item()
                    gp_batch     += gp.item()

                # ---- generator ----
                self.optG.zero_grad()
                noise = torch.randn(bs, self.noiseDim, device=self.device)
                fake = self.genNet(noise)
                fake_noisy = fake + sigma * torch.randn_like(fake, device=self.device)
                loss_G = -self.discNet(fake_noisy).mean()
                loss_G.backward(); self.optG.step()

                # decay noise std
                sigma *= decay

                # schedulers & counters
                self.step_G += 1; self.schedG.step(); self.schedD.step()

                # per‑batch logs
                if self.step_G % 500 == 0:
                    self.GP_log.append(gp_batch / self.nCrit)
                    self.gradG_log.append(param_grad_norm(self.genNet))
                    self.gradD_log.append(param_grad_norm(self.discNet))

                sum_G += loss_G.item(); sum_W += w_dist_batch / self.nCrit
                gp_acc += gp_batch; n_batches += 1

                # mini‑KL
                self.real_buf.append(real.detach().cpu()); self.fake_buf.append(fake.detach().cpu())
                if sum(t.size(0) for t in self.real_buf) >= self.KL_TARGET:
                    r = torch.cat(self.real_buf)[:self.KL_TARGET]
                    f = torch.cat(self.fake_buf)[:self.KL_TARGET]
                    self.KL_log.append(utils.get_kld(r.to(self.device), f.to(self.device)).item())
                    self.real_buf.clear(); self.fake_buf.clear()

            if n_batches == 0: break  # no data processed
            self.G_loss_log.append(sum_G / n_batches)
            self.D_wdist_log.append(-(sum_W / n_batches))
            self.GP_mean_log.append(gp_acc / max(1, n_batches*self.nCrit))

            vG, vD = self._validate()
            self.Val_G_log.append(vG); self.Val_D_log.append(-vD)
            # Save the models if D_wdist_log is lower than previous best
            if self.best_ValD > -vD > 0:
                self.best_ValD = -vD
                print(f"New best Val_D: {self.best_ValD:.4f} at epoch {epoch}")
                torch.save(self.genNet.state_dict(), f"{self.outputDir}{self.dataGroup}_Gen_model.pt")
                torch.save(self.discNet.state_dict(), f"{self.outputDir}{self.dataGroup}_Disc_model.pt")

            # --- early‑stop slopes ---
            win = min(self.SLOPE_WINDOW, len(self.KL_log), len(self.D_wdist_log))
            if win >= 3 and epoch > 15:
                def slope(arr):
                    try:
                        return np.polyfit(range(win), arr[-win:], 1)[0]
                    except np.linalg.LinAlgError:
                        return 0.0
                if abs(slope(self.KL_log)) < self.SLOPE_EPS and abs(slope(self.D_wdist_log)) < self.SLOPE_EPS:
                    print('Early stop: slopes flat'); break



    # --------------------- validate ------------------ #
    def _validate(self):
        self.genNet.eval(); self.discNet.eval(); sG=sD=n=0
        with torch.no_grad():
            for real in self.dl_val:
                real = real.to(self.device); bs = real.size(0)
                loss_real = -self.discNet(real).mean()
                fake = self.genNet(torch.randn(bs, self.noiseDim, device=self.device))
                loss_fake = self.discNet(fake).mean()
                sD += (loss_real + loss_fake).item()
                sG += (-self.discNet(fake).mean()).item(); n += 1
        self.genNet.train(); self.discNet.train(); return sG/n, sD/n