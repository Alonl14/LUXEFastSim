# fastsim_cluster_main.py  –  launcher (inner / outer1 / outer2)
import os, sys, json, time, numpy as np, utils
from trainerFactory import create_trainer

# ------------------------------------------------------------------ #
# 0.  CLI                                                             #
# ------------------------------------------------------------------ #
num_epochs   = int(sys.argv[1])
output_dir   = sys.argv[2]
config_stamp = sys.argv[3]          # e.g. "_cfg_v7.json"

cfg_dir = "/srv01/agrp/alonle/LUXEFastSim/Config"
regions = ["inner", "outer1", "outer2"]

# ------------------------------------------------------------------ #
# 1.  Load configs, patch runtime fields, build trainers              #
# ------------------------------------------------------------------ #
trainers = {}
for reg in regions:
    with open(os.path.join(cfg_dir, f"{reg}{config_stamp}"), "r") as fp:
        cfg = json.load(fp)

    cfg.update({"outputDir": output_dir, "numEpochs": num_epochs})
    trainers[reg] = create_trainer(cfg)

# ------------------------------------------------------------------ #
# 2.  Run: outer regions first, then inner                            #
# ------------------------------------------------------------------ #
t0 = time.localtime();  print(f"Start : {utils.get_time(t0)}")
for reg in ["outer1", "outer2", "inner"]:
    print(f"--- Training {reg} ---")
    trainers[reg].run()
    print(f"{reg} done : {utils.get_time(time.localtime(), t0)}")

# ------------------------------------------------------------------ #
# 3.  Persist logs                                                    #
# ------------------------------------------------------------------ #
log_map = {          # trainer attribute → prefix
    "KL_log"        : "KL",
    "D_wdist_log"   : "D",
    "Val_D_log"     : "ValD",
    "G_loss_log"    : "G",
    "Val_G_log"     : "ValG",
    "GP_log"        : "GP",        # gradient-penalty trace
    "gradG_log"   : "GradG",
    "gradD_log"   : "GradD"
}

for reg, tr in trainers.items():
    for attr, prefix in log_map.items():
        arr = getattr(tr, attr, None)
        if arr is not None and len(arr):
            np.save(os.path.join(tr.outputDir, f"{prefix}_{reg}.npy"), np.asarray(arr))

print("All logs saved – job finished.")
