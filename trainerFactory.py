from sklearn.preprocessing import QuantileTransformer as qt
from dataset import ParticleDataset
from generator import InnerGenerator, OuterGenerator
from discriminator import InnerDiscriminator, OuterDiscriminator
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from trainer import Trainer


def create_trainer(cfg):
    QT_inner = qt(output_distribution='normal', n_quantiles=cfg['nQuantiles'], subsample=cfg['subsample'])
    dataset = ParticleDataset(cfg['data'], cfg['dataNorm'], QT_inner, cfg['dataGroup'])
    dataloader = DataLoader(dataset, batch_size=cfg['batchSize'], shuffle=True)
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

    cfgDict = {
        'genNet': genNet,
        'noiseDim': cfg['noiseDim'],
        'discNet': discNet,
        'outputDir': cfg['outputDir'],
        'dataloader': dataloader,
        'dataset': dataset,
        'dataGroup': cfg['dataGroup'],
        'numEpochs': cfg['numEpochs'],
        'device': device,
        'nCrit': cfg['nCrit'],
        'Lambda': cfg['Lambda']
    }

    genOptimizer = optim.Adam(cfgDict['genNet'].parameters(), lr=0.0001, betas=(0.5, 0.999))
    discOptimizer = optim.Adam(cfgDict['discNet'].parameters(), lr=0.0001, betas=(0.5, 0.999))

    cfgDict['genOptimizer'] = genOptimizer
    cfgDict['discOptimizer'] = discOptimizer

    return Trainer(cfgDict=cfgDict)


class TrainerFactory:
    pass

