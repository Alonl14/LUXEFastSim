import json
import sys
from trainerFactory import create_trainer
import torch
import numpy as np
import time
import utils
import pandas as pd
import pickle

beg_time = time.localtime()
print(f"Starting timer at : {utils.get_time(beg_time)}")
config_dir = "/srv01/agrp/alonle/LUXEFastSim/Config"
config_name = sys.argv[3]

with open(config_dir+"/inner"+config_name, 'r') as inner_file:  #
    cfg_inner = json.loads(inner_file.read())
with open(config_dir+"/outer"+config_name, 'r') as outer_file:
    cfg_outer = json.loads(outer_file.read())

cfg_inner['outputDir'] = sys.argv[2]
cfg_outer['outputDir'] = sys.argv[2]


numEpochs = int(sys.argv[1])

cfg_inner["numEpochs"] = numEpochs
cfg_outer["numEpochs"] = numEpochs

#LOCAL TESTING
# cfg_inner["data_path"] = 'TrainData/neutron_inner_1M.csv'
# cfg_outer["data_path"] = 'TrainData/neutron_outer_5M.csv'


create_time = time.localtime()

print("Creating outer trainer...")
outer_trainer = create_trainer(cfg_outer)
print("Creating inner trainer...")
inner_trainer = create_trainer(cfg_inner)

training_time = time.localtime()
print(f"Trainers Created ! Time elapsed : {utils.get_time(training_time, create_time)} \nStarting outer Training:")

outer_trainer.run()
outer_time = time.localtime()
print(f"Outer Training Done! Time elapsed : {utils.get_time(outer_time, training_time)} \nStarting inner Training:")
inner_trainer.run()
inner_time = time.localtime()
print(f"Inner Training Done! Time elapsed : {utils.get_time(inner_time, outer_time)} \nMaking dataframes:")

KL_in = np.zeros(len(inner_trainer.KL_Div))
KL_out = np.zeros(len(outer_trainer.KL_Div))

for i in range(len(KL_in)):
    KL_in[i] = inner_trainer.KL_Div[i]
for i in range(len(KL_out)):
    KL_out[i] = outer_trainer.KL_Div[i]


np.save(inner_trainer.outputDir+'KL_in.npy', KL_in)
np.save(outer_trainer.outputDir+'KL_out.npy', KL_out)
np.save(inner_trainer.outputDir+'D_losses_in.npy', inner_trainer.D_Losses)
np.save(outer_trainer.outputDir+'D_losses_out.npy', outer_trainer.D_Losses)
