import torch
import numpy as np
import tqdm
import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*You can silence this warning by not passing in num_features.*")


def compute_gradient_penalty(critic, real_samples, fake_samples, lambda_gp, device):
    """
    Compute WGAN-GP gradient penalty.
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty


def corr(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    return (vx * vy).mean() / (vx.std() * vy.std())


class Trainer:
    def __init__(self, cfgDict):
        self.genNet = cfgDict['genNet']
        self.noiseDim = cfgDict['noiseDim']
        self.discNet = cfgDict['discNet']
        self.outputDir = cfgDict['outputDir']
        self.genOptimizer = cfgDict['genOptimizer']
        self.discOptimizer = cfgDict['discOptimizer']
        self.dataloader = cfgDict['dataloader']
        self.valDataloader = cfgDict['valDataloader']  # Validation dataloader
        self.dataset = cfgDict['dataset']
        self.dataGroup = cfgDict['dataGroup']
        self.numEpochs = cfgDict['numEpochs']
        self.device = cfgDict['device']
        self.nCrit = cfgDict['nCrit']
        self.Lambda = cfgDict['Lambda']

        self.trainedGen = torch.Tensor([])
        self.G_Losses = np.array([])
        self.D_Losses = np.array([])
        self.Val_G_Losses = np.array([])  # To store validation generator losses
        self.Val_D_Losses = np.array([])  # To store validation discriminator losses
        self.KL_Div = np.array([])
        self.real_buffer = []
        self.fake_buffer = []
        self.KL_sample_target = 16384

    def run(self):
        print("Starting Training Loop...")

        utils.weights_init(self.genNet)
        utils.weights_init(self.discNet)

        self.genNet.to(self.device)
        self.genNet.train()
        self.discNet.to(self.device)
        self.discNet.train()
        torch.save(self.genNet.state_dict(), self.outputDir + self.dataGroup + '_Gen_model.pt')
        torch.save(self.discNet.state_dict(), self.outputDir + self.dataGroup + '_Disc_model.pt')

        for epoch in tqdm.tqdm(range(self.numEpochs), desc=' epochs', position=0):
            avg_error_G, avg_error_D, currentKLD, iters = 0, 0, 0, 0
            total_batches = len(self.dataloader)
            kl_log_interval = max(1, total_batches // 10)
            for i, data in enumerate(self.dataloader, 0):
                # Training Phase (same as before)
                crit_err_D = 0
                for crit_train in range(self.nCrit):
                    # Train discriminator
                    self.discNet.zero_grad()
                    batch_size = len(data)
                    real_data = data.to(self.device)

                    output = self.discNet(real_data)
                    err_D_real = -torch.mean(output)
                    noise = torch.randn(batch_size, self.noiseDim, device=self.device)
                    fake_p = self.genNet(noise)
                    output = self.discNet(fake_p.detach())
                    err_D_fake = torch.mean(output)

                    gp = compute_gradient_penalty(
                        self.discNet, real_data, fake_p.detach(), self.Lambda, self.device
                    )

                    err_D = err_D_real + err_D_fake + gp
                    err_D.backward()
                    crit_err_D += err_D.item()

                    self.discOptimizer.step()

                # Update generator
                self.genNet.zero_grad()
                output = self.discNet(fake_p)

                err_G = -torch.mean(output)
                err_G.backward()
                self.genOptimizer.step()

                # Calculate losses
                avg_error_G += err_G.item()
                avg_error_D += crit_err_D / self.nCrit

                iters += 1
                # KL divergence calculation if buffer is full enough
                buffer_size = sum(x.size(0) for x in self.real_buffer)
                if buffer_size >= self.KL_sample_target:
                    real_all = torch.cat(self.real_buffer, dim=0)[:self.KL_sample_target]
                    fake_all = torch.cat(self.fake_buffer, dim=0)[:self.KL_sample_target]

                    real_all = real_all.to(self.device)
                    fake_all = fake_all.to(self.device)

                    kl_epoch = utils.get_kld(real_all, fake_all)
                    self.KL_Div = np.append(self.KL_Div, kl_epoch.detach().cpu().numpy())

                    print(f"KL Divergence (Epoch {epoch}): {kl_epoch.item():.4f}")

                    # Reset buffers
                    self.real_buffer = []
                    self.fake_buffer = []
                else:
                    # Store real and fake samples in buffers
                    self.real_buffer.append(real_data.detach().cpu())
                    self.fake_buffer.append(fake_p.detach().cpu())
            # Validation Phase
            val_error_G, val_error_D = self.validate()

            # Save best model based on validation D loss
            if -val_error_D < np.min(self.Val_D_Losses) if len(self.Val_D_Losses) > 0 else float('inf'):
                print("Saving best model based on validation D loss")
                torch.save(self.genNet.state_dict(), self.outputDir + self.dataGroup + '_Gen_model.pt')
                torch.save(self.discNet.state_dict(), self.outputDir + self.dataGroup + '_Disc_model.pt')

            avg_error_G /= iters
            avg_error_D /= iters

            self.G_Losses = np.append(self.G_Losses, avg_error_G)
            self.D_Losses = np.append(self.D_Losses, -avg_error_D)

            self.Val_G_Losses = np.append(self.Val_G_Losses, val_error_G)
            self.Val_D_Losses = np.append(self.Val_D_Losses, -val_error_D)

            print(f'{epoch}/{self.numEpochs}\tLoss_D: {-avg_error_D:.4f}\tLoss_G: {avg_error_G:.4f}')
            print(f'Validation Loss_D: {-val_error_D:.4f}\tValidation Loss_G: {val_error_G:.4f}')

    def validate(self):
        self.genNet.eval()
        self.discNet.eval()

        val_error_G, val_error_D = 0, 0
        with torch.no_grad():  # Disable gradient calculations during validation
            for data in self.valDataloader:
                batch_size = len(data)
                real_data = data.to(self.device)

                # Discriminator validation
                output = self.discNet(real_data)
                err_D_real = -torch.mean(output)

                noise = torch.randn(batch_size, self.noiseDim, device=self.device)
                fake_p = self.genNet(noise)
                output = self.discNet(fake_p)
                err_D_fake = torch.mean(output)

                # No gradient penalty during validation
                err_D = err_D_real + err_D_fake
                val_error_D += err_D.item()

                # Generator validation
                err_G = -torch.mean(output)
                val_error_G += err_G.item()

        val_error_G /= len(self.valDataloader)
        val_error_D /= len(self.valDataloader)

        # Switch back to train mode
        self.genNet.train()
        self.discNet.train()

        return val_error_G, val_error_D
