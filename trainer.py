import torch
import numpy as np
import tqdm
import utils


def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


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
        self.applyQT = cfgDict['applyQT']

        self.trainedGen = torch.Tensor([])
        self.G_Losses = np.array([])
        self.D_Losses = np.array([])
        self.Val_G_Losses = np.array([])  # To store validation generator losses
        self.Val_D_Losses = np.array([])  # To store validation discriminator losses
        self.KL_Div = np.array([])

    def run(self):
        print("Starting Training Loop...")

        utils.weights_init(self.genNet)
        utils.weights_init(self.discNet)

        self.genNet.to(self.device)
        self.genNet.train()
        self.discNet.to(self.device)
        self.discNet.train()

        for epoch in tqdm.tqdm(range(self.numEpochs), desc=' epochs', position=0):
            avg_error_G, avg_error_D, currentKLD, iters = 0, 0, 0, 0
            total_batches = len(self.dataloader)
            kl_log_interval = max(1, total_batches // 10)
            for i, data in tqdm.tqdm(enumerate(self.dataloader, 0), desc=' batch', position=1, leave=False):
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

                    epsilon = torch.rand(1, device=self.device, requires_grad=True)
                    gradient = get_gradient(self.discNet, real_data, fake_p.detach(), epsilon)
                    gradient_norm = gradient.norm(2, dim=1)
                    penalty = self.Lambda * torch.mean((gradient_norm - 1) ** 2)
                    err_D = err_D_real + err_D_fake + penalty
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

                addCurrentKLD = utils.get_kld(real_data, fake_p)
                currentKLD += addCurrentKLD.detach().numpy()

                iters += 1
                if iters % kl_log_interval == 0:  # Record KL_div 10 times per epoch
                    print(f"Iteration #{iters}")
                    self.KL_Div = np.append(self.KL_Div, currentKLD/kl_log_interval)
                    currentKLD = 0

            # Validation Phase
            val_error_G, val_error_D = self.validate()

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
