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
from trainer import Trainer

# FOR LOCAL TESTING
importlib.reload(dataset)
from dataset import ParticleDataset
importlib.reload(generator)
from generator import InnerGenerator, OuterGenerator
importlib.reload(discriminator)
from discriminator import InnerDiscriminator, OuterDiscriminator


def create_trainer(cfg):
    beg_time = time.localtime()
    print(f"Creating trainer at : {utils.get_time(beg_time)}")
    print("making dataset...")

    particle_ds = ParticleDataset(cfg)
    dataset_time = time.localtime()

    print(f"dataset created, time elapsed : {utils.get_time(beg_time, dataset_time)}")
    print("making data loader...")

    dataloader = DataLoader(particle_ds, batch_size=cfg['batchSize'], shuffle=True)
    dataloader_time = time.localtime()

    print(f"data loader created, time elapsed : {utils.get_time(dataloader_time, dataset_time)}")
    device = torch.device(cfg['device'])
    print(f'Using {device} as device')
    if cfg['dataGroup'] == 'inner':
        genNet = InnerGenerator(noiseDim=cfg['noiseDim'])
        discNet = InnerDiscriminator()
    elif cfg['dataGroup'] == 'outer':
        genNet = OuterGenerator(noiseDim=cfg['noiseDim'])
        discNet = OuterDiscriminator()
    else:
        raise Exception("dataGroup must be either inner or outer")

    genNet = nn.DataParallel(genNet)
    discNet = nn.DataParallel(discNet)

    cfgDict = {
        'genNet': genNet,
        'noiseDim': cfg['noiseDim'],
        'discNet': discNet,
        'outputDir': cfg['outputDir'],
        'dataloader': dataloader,
        'dataset': particle_ds,
        'dataGroup': cfg['dataGroup'],
        'numEpochs': cfg['numEpochs'],
        'device': device,
        'nCrit': cfg['nCrit'],
        'Lambda': cfg['Lambda']
    }

    genOptimizer = optim.Adam(cfgDict['genNet'].parameters(), lr=cfg['learningRate'], betas=(0.5, 0.999))
    discOptimizer = optim.Adam(cfgDict['discNet'].parameters(), lr=cfg['learningRate'], betas=(0.5, 0.999))

    cfgDict['genOptimizer'] = genOptimizer
    cfgDict['discOptimizer'] = discOptimizer

    return Trainer(cfgDict=cfgDict)


class TrainerFactory:
    pass
