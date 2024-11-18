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
with open(config_dir+"/outer_1"+config_name, 'r') as outer_file:
    cfg_outer_1 = json.loads(outer_file.read())
with open(config_dir+"/outer_2"+config_name, 'r') as outer_file:
    cfg_outer_2 = json.loads(outer_file.read())

cfg_inner['outputDir'] = sys.argv[2]
cfg_outer_1['outputDir'] = sys.argv[2]
cfg_outer_2['outputDir'] = sys.argv[2]

numEpochs = int(sys.argv[1])

cfg_inner["numEpochs"] = numEpochs
cfg_outer_1["numEpochs"] = numEpochs
cfg_outer_2["numEpochs"] = numEpochs

#LOCAL TESTING
# cfg_inner["data_path"] = 'TrainData/neutron_inner_1M.csv'
# cfg_outer["data_path"] = 'TrainData/neutron_outer_5M.csv'


create_time = time.localtime()

print("Creating outer trainers...")
outer_trainer_1 = create_trainer(cfg_outer_1)
outer_trainer_2 = create_trainer(cfg_outer_2)
print("Creating inner trainer...")
inner_trainer = create_trainer(cfg_inner)

training_time = time.localtime()
print(f"Trainers Created ! Time elapsed : {utils.get_time(training_time, create_time)} \nStarting outer Training:")

outer_trainer_1.run()
outer_trainer_2.run()
outer_time = time.localtime()
print(f"Outer Training Done! Time elapsed : {utils.get_time(outer_time, training_time)} \nStarting inner Training:")
inner_trainer.run()
inner_time = time.localtime()
print(f"Inner Training Done! Time elapsed : {utils.get_time(inner_time, outer_time)} \nMaking dataframes:")

KL_in = np.zeros(len(inner_trainer.KL_Div))
KL_out_1 = np.zeros(len(outer_trainer_1.KL_Div))
KL_out_2 = np.zeros(len(outer_trainer_2.KL_Div))

for i in range(len(KL_in)):
    KL_in[i] = inner_trainer.KL_Div[i]
for i in range(len(KL_out_1)):
    KL_out_1[i] = outer_trainer_1.KL_Div[i]
for i in range(len(KL_out_2)):
    KL_out_2[i] = outer_trainer_2.KL_Div[i]

np.save(inner_trainer.outputDir+'KL_in.npy', KL_in)
np.save(outer_trainer_1.outputDir+'KL_out.npy', KL_out_1)
np.save(outer_trainer_2.outputDir+'KL_out.npy', KL_out_2)
np.save(inner_trainer.outputDir+'D_losses_in.npy', inner_trainer.D_Losses)
np.save(outer_trainer_1.outputDir+'D_losses_out.npy', outer_trainer_1.D_Losses)
np.save(outer_trainer_2.outputDir+'D_losses_out.npy', outer_trainer_2.D_Losses)
np.save(inner_trainer.outputDir+'Val_D_losses_in.npy', inner_trainer.Val_D_Losses)
np.save(outer_trainer_1.outputDir+'Val_D_losses_out.npy', outer_trainer_1.Val_D_Losses)
np.save(outer_trainer_2.outputDir+'Val_D_losses_out.npy', outer_trainer_2.Val_D_Losses)
np.save(inner_trainer.outputDir+'G_losses_in.npy', inner_trainer.G_Losses)
np.save(outer_trainer_1.outputDir+'G_losses_out.npy', outer_trainer_1.G_Losses)
np.save(outer_trainer_2.outputDir+'G_losses_out.npy', outer_trainer_2.G_Losses)
np.save(inner_trainer.outputDir+'Val_G_losses_in.npy', inner_trainer.Val_G_Losses)
np.save(outer_trainer_1.outputDir+'Val_G_losses_out.npy', outer_trainer_1.Val_G_Losses)
np.save(outer_trainer_2.outputDir+'Val_G_losses_out.npy', outer_trainer_2.Val_G_Losses)
