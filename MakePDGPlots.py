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
utils.make_polar_features(full_df)
print(full_df[" pdg"].unique().tolist())
full_df = full_df[full_df[' rx'] <= 4000]
for pdg in full_df[" pdg"].unique().tolist():
    df_pdg = full_df[full_df[" pdg"] == pdg]
    outer1 = df_pdg[(df_pdg[' xx'] >= 500) | (df_pdg[' yy'] >= 500)]
    outer2 = df_pdg[(df_pdg[' xx'] < 500) & (df_pdg[' yy'] < -1700)]
    inner = df_pdg[(df_pdg[' xx'] < 500) & (df_pdg[' yy'] >= -1700) & (df_pdg[' yy'] < 500)]

    outer1.to_csv(f"/storage/agrp/alonle/GAN_InputSample/{pdg}_outer1.csv", index=False)
    outer2.to_csv(f"/storage/agrp/alonle/GAN_InputSample/{pdg}_outer2.csv", index=False)
    inner.to_csv(f"/storage/agrp/alonle/GAN_InputSample/{pdg}_inner.csv", index=False)

print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
