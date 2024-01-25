import json
import sys
from trainerFactory import create_trainer
import torch
import numpy as np

config_dir = "/srv01/agrp/alonle/LUXEFastSim/Config"

with open(config_dir+"/cfg_inner_cluster.json", 'r') as inner_file:
    cfg_inner = json.loads(inner_file.read())
with open(config_dir+"/cfg_outer_cluster.json", 'r') as outer_file:
    cfg_outer = json.loads(outer_file.read())

#LOCAL TESTING
# cfg_inner["data"] = 'TrainData/neutron_inner_1M.csv'
# cfg_outer["data"] = 'TrainData/neutron_outer_5M.csv'

numEpochs = np.int64(sys.argv[0])

cfg_inner["numEpochs"] = numEpochs
cfg_outer["numEpochs"] = numEpochs

inner_trainer = create_trainer(cfg_inner)
outer_trainer = create_trainer(cfg_outer)

inner_trainer.run()
outer_trainer.run()
KL_in = np.zeros(len(inner_trainer.KL_Div))
KL_out = np.zeros(len(outer_trainer.KL_Div))

for i in range(len(KL_in)):
    KL_in[i] = inner_trainer.KL_Div[i]
    KL_out[i] = outer_trainer.KL_Div[i]

np.save(inner_trainer.outputDir+'/KL_in.npy', KL_in)
np.save(outer_trainer.outputDir+'/KL_out.npy', KL_out)
np.save(inner_trainer.outputDir+'/D_losses_in.npy', inner_trainer.D_Losses)
np.save(outer_trainer.outputDir+'/D_losses_out.npy', outer_trainer.D_Losses)