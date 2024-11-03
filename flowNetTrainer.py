import torch
import numpy as np
import tqdm
import utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning,
                        message=".*You can silence this warning by not passing in num_features.*")


class Trainer:
    def __init__(self, cfgDict):
        self.genNet = cfgDict['genNet']
        self.outputDir = cfgDict['outputDir']
        self.genOptimizer = cfgDict['genOptimizer']
        self.dataloader = cfgDict['dataloader']
        self.valDataloader = cfgDict['valDataloader']  # Validation dataloader
        self.dataset = cfgDict['dataset']
        self.dataGroup = cfgDict['dataGroup']
        self.numEpochs = cfgDict['numEpochs']
        self.device = cfgDict['device']
        self.applyQT = cfgDict['applyQT']
        self.numFeatures = cfgDict['numFeatures']
        self.sigmaMin = cfgDict['sigmaMin']

        self.trainedNet = torch.Tensor([])
        self.Losses = np.array([])
        self.Val_Losses = np.array([])  # To store validation losses
        self.KL_Div = np.array([])

    def run(self):
        print("Starting Training Loop...")

        utils.weights_init(self.genNet)

        self.genNet.to(self.device)
        self.genNet.train()

        for epoch in tqdm.tqdm(range(self.numEpochs), desc=' epochs', position=0):
            avg_loss, iters = 0, 0
            total_batches = len(self.dataloader)
            kl_log_interval = max(1, total_batches // 10)
            for i, data in enumerate(self.dataloader, 0):
                batch_size = len(data)
                # Training Phase
                real_data = data.to(self.device)
                x0 = torch.randn(batch_size, self.numFeatures, device=self.device)
                t = 1-torch.rand(batch_size)  # to be in (0,1]
                t = t.to(self.device)
                psi_t = torch.zeros_like(x0)
                for j in range(batch_size):
                    psi_t[j, :] = (1-(1-self.sigmaMin)*t[j])*x0[j, :] + t[j]*real_data[j, :]

                # Update generator
                self.genNet.zero_grad()
                # psi_t shape is (batch_s, n_f) , t needs to be shape (batch_s, 1)
                # concat along feature axis (dim: 1) results in shape (batch_s, n_f+1)
                output = self.genNet(torch.cat((psi_t, t.reshape(batch_size, 1)), 1))
                loss = torch.norm(output-(real_data-(1-self.sigmaMin)*x0))**2/batch_size
                loss.backward()
                self.genOptimizer.step()

                # Calculate losses
                avg_loss += loss.item()

                iters += 1
                if iters % kl_log_interval == 0:  # Record KL_div 10 times per epoch
                    print(f"Iteration #{iters}")
                    addCurrentKLD = utils.get_kld(real_data,
                                                  self.genNet(torch.cat((x0,
                                                                         torch.ones(batch_size, 1,
                                                                                    device=self.device)),
                                                                        1)))
                    self.KL_Div = np.append(self.KL_Div, addCurrentKLD.detach().numpy())

            # Validation Phase
            val_error = self.validate()

            torch.save(self.genNet.state_dict(), self.outputDir + self.dataGroup + '_Flow_model.pt')

            avg_loss /= iters

            self.Losses = np.append(self.Losses, avg_loss)

            self.Val_Losses = np.append(self.Val_Losses, val_error)

            print(f'{epoch}/{self.numEpochs}\tLoss: {avg_loss:.4f}\tValidation Loss: {val_error:.4f}')

    def validate(self):
        self.genNet.eval()

        val_error_G, val_error_D = 0, 0
        with torch.no_grad():  # Disable gradient calculations during validation
            avg_loss = 0
            for data in self.valDataloader:
                batch_size = len(data)
                real_data = data.to(self.device)

                x0 = torch.randn(batch_size, self.numFeatures, device=self.device)
                t = 1-torch.rand(batch_size)  # to be in (0,1]
                t = t.to(self.device)
                psi_t = torch.zeros_like(x0)
                for j in range(batch_size):
                    psi_t[j, :] = (1-(1-self.sigmaMin)*t[j])*x0[j, :] + t[j]*real_data[j, :]

                output = self.genNet(torch.cat((psi_t, t.reshape(batch_size, 1)), 1))
                loss = torch.norm(output - (real_data - (1 - self.sigmaMin) * x0)) ** 2
                avg_loss += loss.item()

        avg_loss /= len(self.valDataloader)

        # Switch back to train mode
        self.genNet.train()

        return avg_loss
