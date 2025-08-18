# trainer.py  –  WGAN(-GP/R1) trainer with critic terminology, stable metrics, and clean checkpoints

import os
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
#  gradient penalties                                                 #
# ------------------------------------------------------------------ #

def compute_gradient_penalty(critic, real, fake, lam, device):
    """WGAN-GP: ||∇_x D(α x_real + (1-α) x_fake)||2 → 1."""
    b = real.size(0)
    alpha = torch.rand(b, 1, device=device).expand_as(real)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    d_inter = critic(inter)
    grad = torch.autograd.grad(d_inter, inter, torch.ones_like(d_inter, device=device),
                               create_graph=True, retain_graph=True)[0]
    grad_norm = grad.view(b, -1).norm(2, dim=1)
    return lam * ((grad_norm - 1.0) ** 2).mean()


def compute_r1_gradient_penalty(critic, real, lam):
    """R1 regularizer on reals: (1/2) * λ * E[||∇_x D(real)||^2]."""
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
    SLOPE_EPS = 2e-5
    SLOPE_WINDOW = 12  # epochs

    def __init__(self, cfg):
        # networks & data
        self.gen = cfg['genNet']
        self.critic = cfg['discNet']  # kept key name in factory; internally we use .critic
        self.dl_train = cfg['dataloader']
        self.dl_val = cfg['valDataloader']
        self.dataGroup = cfg['dataGroup']

        # hparams
        self.noiseDim = int(cfg['noiseDim'])
        self.numEpochs = int(cfg['numEpochs'])
        self.device = cfg['device'] if isinstance(cfg['device'], torch.device) else torch.device(str(cfg['device']))
        self.nCrit = int(cfg['nCrit'])
        self.Lambda = float(cfg['Lambda'])
        self.gradMetric = cfg.get('gradMetric', 'norm')  # 'norm' (=WGAN-GP) or 'r1'
        self.GMaxSteps = cfg.get('GMaxSteps', None)

        # pre-trained?
        self.trained = (cfg.get('genStateDict') is not None) and (cfg.get('discStateDict') is not None)

        # optimizers
        self.optG = cfg['genOptimizer']
        self.optD = cfg['discOptimizer']

        # IO
        self.outputDir = cfg['outputDir']
        os.makedirs(self.outputDir, exist_ok=True)

        # model selection
        self.best_val_W = float('inf')  # minimize Wasserstein distance (generator’s objective)

        # logs (renamed to critic terminology)
        self.G_loss_log = []
        self.Critic_wdist_log = []       # train estimate of E[D(real)] - E[D(fake)]
        self.Val_G_loss_log = []
        self.Val_Critic_wdist_log = []
        self.KL_log = []
        self.GP_log = []
        self.GP_mean_log = []
        self.gradG_log = []
        self.gradCritic_log = []

        # KL buffers
        self.real_buf, self.fake_buf = [], []
        self.KL_TARGET = 8192

        # LR schedulers: warmup G only (critic stays constant by default)
        self.step_G = 0
        self.schedG = LambdaLR(self.optG, lr_lambda=self._warmup)
        self.schedD = None  # optional; can add a scheduler if needed

        # instance noise schedule
        self.sigma0 = float(cfg.get("instanceNoiseSigma", 1e-2))
        self.noise_decay = float(cfg.get("instanceNoiseDecay", 0.9998))
        self.sigma_min = float(cfg.get("instanceNoiseMin", 1e-4))
        self._sigma = self.sigma0

    # --------------------- utils --------------------- #
    def _warmup(self, step):
        step = min(step, self.WARMUP_STEPS)
        return 0.5 * (1.0 - np.cos(step / self.WARMUP_STEPS * np.pi))

    def _add_instance_noise(self, x):
        if self._sigma <= 0.0:
            return x
        return x + self._sigma * torch.randn_like(x, device=self.device)

    # --------------------- main ---------------------- #
    def run(self):
        # init weights only if not loaded from state dict
        if not self.trained:
            print("[trainer] Initializing weights…")
            utils.weights_init(self.gen)
            utils.weights_init(self.critic)

        self.gen.to(self.device).train()
        self.critic.to(self.device).train()

        stop_by_steps = False

        for epoch in tqdm.tqdm(range(self.numEpochs), desc='epochs'):
            if stop_by_steps:
                break

            sum_G = 0.0
            sum_W = 0.0
            gp_acc = 0.0
            n_batches = 0

            for real in self.dl_train:
                if isinstance(real, np.ndarray):
                    real = torch.from_numpy(real)
                real = real.to(self.device)

                if self.GMaxSteps is not None and self.step_G >= int(self.GMaxSteps):
                    stop_by_steps = True
                    break

                bs = real.size(0)

                # ---- critic steps ----
                wdist_epoch_batch = 0.0
                gp_epoch_batch = 0.0
                for _ in range(self.nCrit):
                    self.optD.zero_grad(set_to_none=True)

                    # WGAN critic aims to maximize E[D(real)] - E[D(fake)]
                    real_in = real
                    if self.gradMetric == 'norm':
                        # WGAN-GP allows instance noise on both streams
                        real_in = self._add_instance_noise(real_in)

                    d_real = self.critic(real_in).mean()
                    noise = torch.randn(bs, self.noiseDim, device=self.device)
                    fake = self.gen(noise)
                    fake_in = fake
                    # instance noise on fakes is OK in both modes
                    fake_in = self._add_instance_noise(fake_in)

                    d_fake = self.critic(fake_in.detach()).mean()
                    w = (d_real - d_fake)  # positive Wasserstein estimate

                    if self.gradMetric == 'norm':
                        gp = compute_gradient_penalty(self.critic, real_in, fake_in.detach(), self.Lambda, self.device)
                    else:  # 'r1'
                        # R1 is defined on clean reals; we didn’t add noise above in this branch
                        gp = compute_r1_gradient_penalty(self.critic, real, self.Lambda)

                    # maximize w  ↔  minimize -(w) ; so critic loss:
                    loss_D = -(w) + gp
                    loss_D.backward()
                    self.optD.step()

                    wdist_epoch_batch += w.item()
                    gp_epoch_batch += gp.item()

                # ---- generator step ----
                self.optG.zero_grad(set_to_none=True)
                noise = torch.randn(bs, self.noiseDim, device=self.device)
                fake = self.gen(noise)
                fake_in = self._add_instance_noise(fake)
                loss_G = -self.critic(fake_in).mean()  # minimize to reduce Wasserstein distance
                loss_G.backward()
                self.optG.step()

                # decay instance noise std per step with a floor
                self._sigma = max(self._sigma * self.noise_decay, self.sigma_min)

                # schedulers & counters
                self.step_G += 1
                self.schedG.step()
                if self.schedD is not None:
                    self.schedD.step()

                # per-batch logs (sparse)
                if self.step_G % 500 == 0:
                    self.GP_log.append(gp_epoch_batch / self.nCrit)
                    self.gradG_log.append(param_grad_norm(self.gen))
                    self.gradCritic_log.append(param_grad_norm(self.critic))

                sum_G += loss_G.item()
                sum_W += (wdist_epoch_batch / self.nCrit)
                gp_acc += gp_epoch_batch
                n_batches += 1

                # mini-KL
                self.real_buf.append(real.detach().cpu())
                self.fake_buf.append(fake.detach().cpu())
                if sum(t.size(0) for t in self.real_buf) >= self.KL_TARGET:
                    r = torch.cat(self.real_buf)[:self.KL_TARGET]
                    f = torch.cat(self.fake_buf)[:self.KL_TARGET]
                    self.KL_log.append(utils.get_kld(r.to(self.device), f.to(self.device)).item())
                    self.real_buf.clear()
                    self.fake_buf.clear()

            if n_batches == 0:
                break  # no data processed this epoch

            # epoch logs
            self.G_loss_log.append(sum_G / n_batches)
            self.Critic_wdist_log.append(sum_W / n_batches)  # positive estimate
            self.GP_mean_log.append(gp_acc / max(1, n_batches * self.nCrit))

            # validation
            vG, vW = self._validate()
            self.Val_G_loss_log.append(vG)
            self.Val_Critic_wdist_log.append(vW)  # we minimize this for model selection

            # model selection: minimize val Wasserstein distance
            if vW < self.best_val_W:
                self.best_val_W = vW
                print(f"[trainer] New best Val W: {self.best_val_W:.6f} (epoch {epoch})")
                gen_path = os.path.join(self.outputDir, f"{self.dataGroup}_Gen_model.pt")
                crit_path = os.path.join(self.outputDir, f"{self.dataGroup}_Critic_model.pt")
                torch.save(self.gen.state_dict(), gen_path)
                torch.save(self.critic.state_dict(), crit_path)

            # --- optional early stop by flat slopes ---
            win = min(self.SLOPE_WINDOW, len(self.KL_log), len(self.Critic_wdist_log))
            if win >= 3 and epoch > 60:
                def slope(arr):
                    try:
                        return np.polyfit(range(win), arr[-win:], 1)[0]
                    except np.linalg.LinAlgError:
                        return 0.0

                if abs(slope(self.KL_log)) < self.SLOPE_EPS and abs(slope(self.Critic_wdist_log)) < self.SLOPE_EPS:
                    print('[trainer] Early stop: slopes flat')
                    break

    # --------------------- validate ------------------ #
    def _validate(self):
        """Return (gen_loss_val, wasserstein_val)."""
        self.gen.eval()
        self.critic.eval()
        sG = 0.0
        sW = 0.0
        n = 0
        with torch.no_grad():
            for real in self.dl_val:
                if isinstance(real, np.ndarray):
                    real = torch.from_numpy(real)
                real = real.to(self.device)
                bs = real.size(0)

                d_real = self.critic(real).mean()
                fake = self.gen(torch.randn(bs, self.noiseDim, device=self.device))
                d_fake = self.critic(fake).mean()

                # generator validation loss (no noise)
                g_loss = -d_fake
                w = (d_real - d_fake)

                sG += g_loss.item()
                sW += w.item()
                n += 1

        self.gen.train()
        self.critic.train()
        n = max(1, n)
        return sG / n, sW / n
