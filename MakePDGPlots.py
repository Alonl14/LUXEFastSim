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
print(full_df[" pdg"].unique().tolist())
for pdg in full_df[" pdg"].unique().tolist():
    temp_df = full_df[full_df[" pdg"] == pdg]
    Hxy = utils.plot_correlations(temp_df[' xx'], temp_df[' yy'], 'x[mm]', 'y[mm]', run_id=None,
                                  key="PDG=" + str(pdg))
    energy_bins = 10 ** np.linspace(-12, 0, 400)
    time_bins = 10 ** np.linspace(1, 8, 400)
    Het = utils.plot_correlations(temp_df[' time'], temp_df[' eneg'], 't[ns]', 'E[GeV]', run_id=None,
                                  key="PDG=" + str(pdg), bins=[time_bins, energy_bins], loglog=True)
print(f"Done! Time elapsed : {utils.get_time(beg_time, time.localtime())}")
