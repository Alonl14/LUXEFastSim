"""
trainer.py – revised (explicit attrs, internal grad‑norm, working validate)
--------------------------------------------------------------------------
Key fixes since last version
  • Validation loop rewritten – now returns correct mean W‑distance
  • Minor tidy‑ups: removed unused args, added type hints

This is still a drop‑in replacement; public API unchanged.
"""

import torch
import numpy as np
import tqdm
import utils
import warnings
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*does not have valid feature names.*")
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message=".*You can silence this warning by not passing in num_features.*")

# ------------------------------------------------------------------ #
#  Helpers                                                           #
# ------------------------------------------------------------------ #

def compute_gradient_penalty(critic, real_samples, fake_samples, lambda_gp, device):
    """WGAN‑GP gradient penalty."""
    batch = real_samples.size(0)
    alpha = torch.rand(batch, 1, device=device).expand_as(real_samples)
    inter = alpha * real_samples + (1 - alpha) * fake_samples
    inter.requires_grad_(True)

    d_inter = critic(inter)
    grad = torch.autograd.grad(
        outputs=d_inter, inputs=inter,
        grad_outputs=torch.ones_like(d_inter, device=device),
        create_graph=True, retain_graph=True)[0]

    grad_norm = grad.view(batch, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


def param_grad_norm(net):
    """L2 norm of parameter gradients; returns 0 if grads not yet computed."""
    sq_sum = 0.0
    for p in net.parameters():
        if p.grad is not None:
            sq_sum += p.grad.detach().norm(2).item() ** 2
    return sq_sum ** 0.5

# ------------------------------------------------------------------ #
#  Trainer                                                           #
# ------------------------------------------------------------------ #

class Trainer:
    """Single‑GPU WGAN‑GP trainer (IDE‑friendly attributes)."""

    WARMUP_STEPS = 1_000
    KL_STOP_STD = 0.01
    SLOPE_EPS = 0.002
    SLOPE_WINDOW = 12  # epochs

    def __init__(self, cfgDict: dict):
        # architecture & data
        self.genNet = cfgDict['genNet']
        self.discNet = cfgDict['discNet']
        self.dl_train = cfgDict['dataloader']
        self.dl_val = cfgDict['valDataloader']
        self.dataGroup = cfgDict['dataGroup']

        # hparams
        self.noiseDim = cfgDict['noiseDim']
        self.numEpochs = cfgDict['numEpochs']
        self.device = cfgDict['device']
        self.nCrit = cfgDict['nCrit']
        self.Lambda = cfgDict['Lambda']

        self.optG = cfgDict['genOptimizer']
        self.optD = cfgDict['discOptimizer']
        self.outDir = cfgDict['outputDir']

        # logs
        self.G_loss_log, self.D_wdist_log = [], []
        self.Val_G_log, self.Val_D_log = [], []
        self.KL_log, self.GP_log = [], []
        self.gradG_log, self.gradD_log = [], []

        # mini‑buffers for KL
        self.real_buf, self.fake_buf = [], []
        self.KL_TARGET = 8192

        # LR warm‑up schedulers
        self.step_G = 0
        self.schedG = LambdaLR(self.optG, lr_lambda=self._warmup_scale)
        self.schedD = LambdaLR(self.optD, lr_lambda=self._warmup_scale)

    # -------------------------------------------------------------- #
    #  Utilities                                                     #
    # -------------------------------------------------------------- #

    def _warmup_scale(self, step: int):
        step = min(step, self.WARMUP_STEPS)
        return 0.5 * (1 - np.cos(step / self.WARMUP_STEPS * np.pi))

    # -------------------------------------------------------------- #
    #  Main loop                                                     #
    # -------------------------------------------------------------- #

    def run(self):
        print("Starting Training Loop…")
        utils.weights_init(self.genNet)
        utils.weights_init(self.discNet)
        self.genNet.to(self.device).train()
        self.discNet.to(self.device).train()

        torch.save(self.genNet.state_dict(), f"{self.outDir}{self.dataGroup}_Gen_model.pt")
        torch.save(self.discNet.state_dict(), f"{self.outDir}{self.dataGroup}_Disc_model.pt")

        for epoch in tqdm.tqdm(range(self.numEpochs), desc="epochs"):
            sum_G, sum_D, n_batches = 0.0, 0.0, 0

            for real in self.dl_train:
                real = real.to(self.device)
                bs = real.size(0)

                # ---- critic ----
                crit_loss = 0.0
                for _ in range(self.nCrit):
                    self.optD.zero_grad()
                    loss_real = -self.discNet(real).mean()

                    noise = torch.randn(bs, self.noiseDim, device=self.device)
                    fake = self.genNet(noise)
                    loss_fake = self.discNet(fake.detach()).mean()

                    gp = compute_gradient_penalty(self.discNet, real, fake.detach(), self.Lambda, self.device)
                    loss_D = loss_real + loss_fake + gp
                    loss_D.backward()
                    self.optD.step()
                    crit_loss += loss_D.item()

                # ---- generator ----
                self.optG.zero_grad()
                noise = torch.randn(bs, self.noiseDim, device=self.device)
                fake = self.genNet(noise)
                loss_G = -self.discNet(fake).mean()
                loss_G.backward()
                self.optG.step()

                # ---- LR schedulers ----
                self.step_G += 1
                self.schedG.step(); self.schedD.step()

                # ---- logs ----
                self.GP_log.append(gp.item())
                self.gradG_log.append(param_grad_norm(self.genNet))
                self.gradD_log.append(param_grad_norm(self.discNet))

                sum_G += loss_G.item(); sum_D += crit_loss / self.nCrit; n_batches += 1

                # ---- mini‑KL ----
                self.real_buf.append(real.detach().cpu())
                self.fake_buf.append(fake.detach().cpu())
                if sum(t.size(0) for t in self.real_buf) >= self.KL_TARGET:
                    r = torch.cat(self.real_buf)[:self.KL_TARGET]
                    f = torch.cat(self.fake_buf)[:self.KL_TARGET]
                    kl_val = utils.get_kld(r.to(self.device), f.to(self.device)).item()
                    self.KL_log.append(kl_val)
                    self.real_buf.clear(); self.fake_buf.clear()

            # ---- epoch stats ----
            self.G_loss_log.append(sum_G / n_batches)
            self.D_wdist_log.append(-(sum_D / n_batches))

            vG, vD = self._validate()
            self.Val_G_log.append(vG); self.Val_D_log.append(-vD)

            print(f"{epoch}/{self.numEpochs}\tW_dist:{-sum_D/n_batches:.4f}\tG:{sum_G/n_batches:.4f}")
            print(f"\tVal_W_dist:{-vD:.4f}\tVal_G:{vG:.4f}")

            # ---- rolling slope early stop ----
            win = min(self.SLOPE_WINDOW, len(self.KL_log), len(self.D_wdist_log))
            if win >= 3 and epoch > 15:
                kl_slope = np.polyfit(range(win), self.KL_log[-win:], 1)[0]
                w_slope  = np.polyfit(range(win), self.D_wdist_log[-win:], 1)[0]
                if abs(kl_slope) < self.SLOPE_EPS and abs(w_slope) < self.SLOPE_EPS:
                    print("Rolling slopes flat – early stop.")
                    break

    # -------------------------------------------------------------- #
    #  Validation                                                    #
    # -------------------------------------------------------------- #

    def _validate(self):
        """Return (gen_loss, critic_loss) averaged over validation set."""
        self.genNet.eval(); self.discNet.eval()
        sum_G, sum_D, n = 0.0, 0.0, 0
        with torch.no_grad():
            for real in self.dl_val:
                real = real.to(self.device)
                bs = real.size(0)

                # critic (no GP)
                loss_real = -self.discNet(real).mean()
                noise = torch.randn(bs, self.noiseDim, device=self.device)
                fake = self.genNet(noise)
                loss_fake = self.discNet(fake).mean()
                loss_D = loss_real + loss_fake

                # generator
                loss_G = -self.discNet(fake).mean()

                sum_G += loss_G.item(); sum_D += loss_D.item(); n += 1

        self.genNet.train(); self.discNet.train()
        return sum_G / n, sum_D / n
