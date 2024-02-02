import torch
import numpy as np
import pandas as pd
import tqdm
import utils
from numpy import random


def get_gradient(crit, real, fake,
                 epsilon):
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
        self.dataset = cfgDict['dataset']
        self.dataGroup = cfgDict['dataGroup']
        self.numEpochs = cfgDict['numEpochs']
        self.device = cfgDict['device']
        self.nCrit = cfgDict['nCrit']
        self.Lambda = cfgDict['Lambda']

        self.trainedGen = torch.Tensor([])
        self.G_Losses = np.array([])
        self.D_Losses = np.array([])
        self.KL_Div = np.array([])
    def run(self):
        generated_df = pd.DataFrame([])

        print("Starting Training Loop...")

        utils.weights_init(self.genNet)
        utils.weights_init(self.discNet)

        self.genNet.to(self.device)
        self.genNet.train()
        self.discNet.to(self.device)
        self.discNet.train()

        iters = 0
        for epoch in tqdm.tqdm_notebook(range(self.numEpochs), desc=' epochs', position=0):

            avg_error_G, avg_error_D, currentKLD = 0, 0, 0

            for i, data in tqdm.tqdm_notebook(enumerate(self.dataloader, 0), desc=' batch', position=1, leave=False):
                # Update the discriminator network

                crit_err_D = 0

                for crit_train in range(self.nCrit):
                    # Train with all-real batch
                    self.discNet.zero_grad()
                    b_size = len(data)
                    real_data = data.to(self.device)

                    output = self.discNet(real_data)

                    err_D_real = -torch.mean(output)

                    # Train with all-fake batch
                    noise = torch.randn(b_size, self.noiseDim, device=self.device)
                    fake_p = self.genNet(noise)

                    output = self.discNet(fake_p.detach())
                    err_D_fake = torch.mean(output)
                    fake_p.to(self.device)

                    epsilon = torch.rand(1, device=self.device, requires_grad=True)
                    gradient = get_gradient(self.discNet, real_data, fake_p.detach(), epsilon)
                    gradient_norm = gradient.norm(2, dim=1)
                    penalty = self.Lambda * torch.mean(torch.norm(gradient_norm - 1))

                    err_D = err_D_real + err_D_fake + penalty
                    err_D.backward()
                    crit_err_D += err_D.item()

                    # update the discriminator network
                    self.discOptimizer.step()

                # Update the Generator network
                self.genNet.zero_grad()
                output = self.discNet(fake_p)
                err_G = -torch.mean(output)
                err_G.backward()

                # update the generator network
                self.genOptimizer.step()

                # computing the average losses and discriminator
                avg_error_G += err_G.item()
                avg_error_D += crit_err_D/self.nCrit

                addCurrentKLD = utils.get_kld(real_data, fake_p)
                addCurrentKLD = addCurrentKLD.detach().numpy()
                currentKLD += addCurrentKLD

                if iters % 100 == 0:
                    print("Iteration #"+str(iters))
                    self.KL_Div = np.append(self.KL_Div, currentKLD/100)
                    currentKLD = 0
                iters += 1

            if len(self.D_Losses) > 0:
                if avg_error_D < self.D_Losses[-1]:
                    torch.save(self.genNet.state_dict(), self.outputDir+self.dataGroup+'_Gen_model.pt')
            else:
                torch.save(self.genNet.state_dict(), self.outputDir + self.dataGroup + '_Gen_model.pt')


            avg_error_G = avg_error_G/iters
            self.G_Losses = np.append(self.G_Losses, avg_error_G)
            avg_error_D = avg_error_D/iters
            self.D_Losses = np.append(self.D_Losses, avg_error_D)
            print(f'{epoch}/{self.numEpochs}\tLoss_D: {avg_error_D:.4f}\tLoss_G: {avg_error_G:.4f}')
        return generated_df
