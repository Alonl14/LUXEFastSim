import sys
import time
import utils
import pandas as pd

beg_time = time.localtime()
print(f"Starting timer at : {utils.get_time(beg_time)}")
full_data_path = "/storage/agrp/alonle/GAN_InputSample/LUXEDumpFiles_FullSim_0p06BX_DetId33_ForStudents.csv"
full_df = pd.read_csv(full_data_path)
for pdg in full_df[" pdg"].unique().tolist():
    GHxy, GHet, GHrth, GHpp = utils.make_plots(full_df[full_df[" pdg"] == pdg], "outer",
                                               key="PDG=" + str(pdg))
print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
