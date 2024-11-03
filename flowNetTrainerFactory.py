import dataset
import importlib
import generator
import discriminator
import time
import utils
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
# FOR LOCAL TESTING
importlib.reload(dataset)
from dataset import ParticleDataset
import flowNet
importlib.reload(flowNet)
from flowNet import FlowNet
import flowNetTrainer
importlib.reload(flowNetTrainer)
from flowNetTrainer import Trainer
from torch.utils.data import random_split


def create_flow_trainer(cfg):
    beg_time = time.localtime()
    print(f"Creating trainer at : {utils.get_time(beg_time)}")
    print("making dataset...")

    particle_ds = ParticleDataset(cfg)
    utils.save_cfg(cfg)

    dataset_time = time.localtime()

    print(f"dataset created, time elapsed : {utils.get_time(beg_time, dataset_time)}")
    print("splitting dataset into train and validation sets...")

    val_size = int(0.2 * len(particle_ds))  # 20% for validation
    train_size = len(particle_ds) - val_size
    train_ds, val_ds = random_split(particle_ds, [train_size, val_size])

    print("making data loaders...")

    dataloader = DataLoader(train_ds, batch_size=cfg['batchSize'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=cfg['batchSize'], shuffle=False)

    dataloader_time = time.localtime()

    print(f"data loaders created, time elapsed : {utils.get_time(dataloader_time, dataset_time)}")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = torch.device(cfg['device'])
    print(f'Using {device} as device')

    numFeatures = len(cfg["features"].keys())

    # For the inverseQT sampling to work we need :
    if not cfg['applyQT']:
        cfg['noiseDim'] = numFeatures

    genNet = FlowNet(numFeatures=numFeatures)

    genNet = nn.DataParallel(genNet)

    cfgDict = {
        'genNet': genNet,
        'noiseDim': cfg['noiseDim'],
        'outputDir': cfg['outputDir'],
        'dataloader': dataloader,
        'valDataloader': val_dataloader,  # Add validation dataloader
        'dataset': particle_ds,
        'dataGroup': cfg['dataGroup'],
        'numEpochs': cfg['numEpochs'],
        'device': device,
        'nCrit': cfg['nCrit'],
        'Lambda': cfg['Lambda'],
        'applyQT': cfg['applyQT'],
        'sigmaMin': 10**-5,
        'numFeatures': numFeatures
    }

    genOptimizer = optim.Adam(cfgDict['genNet'].parameters(), lr=cfg['criticLearningRate'], betas=(0.5, 0.999))

    cfgDict['genOptimizer'] = genOptimizer

    return Trainer(cfgDict=cfgDict)


class FlowNetTrainerFactory:
    pass
