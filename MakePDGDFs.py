import sys
import time
import utils
import pandas as pd
import numpy as np

beg_time = time.localtime()
print(f"Starting timer at : {utils.get_time(beg_time)}")
full_data_path = "/storage/agrp/alonle/GAN_InputSample/LUXEDumpFiles_FullSim_0p06BX_DetId33_ForStudents.csv"
path = "/storage/agrp/alonle/GAN_Output"
full_df = pd.read_csv(full_data_path)
pdg_dict = {11: "Electron",
            -11: "Positron",
            12: "Neutrino",
            -12: "Anti-Neutrino",
            13: "Muon",
            -13: "Anti-Muon",
            14: "Muon-Neutrino",
            -14: "Anti-Muon-Neutrino",
            22: "Photon",
            130: "K0-L",
            211: "Pion+",
            -211: "Pion-",
            2112: "Neutron",
            2212: "Proton",
            1000010020: "Deuteron",
            1000020040: "Alpha",
            1000030070: "Li-7",
            1000060120: "C-12",
            1000080160: "O-16",
            1000260540: "Fe-54"
            }

print(full_df[" pdg"].unique().tolist())
for pdg in full_df[" pdg"].unique().tolist():
    temp_df = full_df[full_df[" pdg"] == pdg]
print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
