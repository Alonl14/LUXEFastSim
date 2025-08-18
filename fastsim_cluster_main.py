# fastsim_cluster_main.py – launcher (inner / outer1 / outer2) with argparse & critic logs
import os
import sys
import json
import time
import argparse
import numpy as np

import utils
from trainerFactory import create_trainer


def parse_args():
    p = argparse.ArgumentParser(description="Run FastSim training for multiple regions.")
    p.add_argument("num_epochs", type=int, help="Number of epochs to train each region.")
    p.add_argument("output_dir", type=str, help="Directory to write checkpoints and logs.")
    p.add_argument("config_stamp", type=str, help='Config suffix (e.g., "_cfg_v7.json").')
    p.add_argument("--cfg_dir", type=str,
                   default="/srv01/agrp/alonle/LUXEFastSim/Config",
                   help="Directory containing region configs.")
    p.add_argument("--regions", type=str, nargs="+",
                   default=["outer1", "outer2", "inner"],
                   help="Regions to train in order.")
    p.add_argument("--seed", type=int, default=1337, help="Seed for split/repro.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Region order
    regions = list(args.regions)

    # Start banner
    t0 = time.localtime()
    print(f"Start : {utils.get_time(t0)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Config dir: {args.cfg_dir}")
    print(f"Regions   : {regions}")

    # Build trainers per region
    trainers = {}
    for reg in regions:
        cfg_path = os.path.join(args.cfg_dir, f"{reg}{args.config_stamp}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing config for region '{reg}': {cfg_path}")

        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        # Patch runtime fields
        cfg.update({
            "outputDir": args.output_dir,
            "numEpochs": args.num_epochs,
            "seed": args.seed,
        })

        trainers[reg] = create_trainer(cfg)

    # Train in requested order
    for reg in regions:
        print(f"--- Training {reg} ---")
        t_reg0 = time.localtime()
        trainers[reg].run()
        print(f"{reg} done : {utils.get_time(time.localtime(), t_reg0)} (since start {utils.get_time(time.localtime(), t0)})")

    # Persist logs
    # NOTE: filenames kept for backward compatibility with existing analysis scripts in utils.check_run
    # Map Trainer attribute → filename prefix
    log_map = {
        "KL_log"               : "KL",
        "Critic_wdist_log"     : "D",      # formerly D_wdist_log
        "Val_Critic_wdist_log" : "ValD",   # formerly Val_D_log
        "G_loss_log"           : "G",
        "Val_G_loss_log"       : "ValG",
        "GP_log"               : "GP",
        "gradG_log"            : "GradG",
        "gradCritic_log"       : "GradD",  # formerly gradD_log
    }

    for reg, tr in trainers.items():
        for attr, prefix in log_map.items():
            arr = getattr(tr, attr, None)
            if arr is None:
                continue
            # convert to np array if it isn't one already
            try:
                np_arr = np.asarray(arr)
            except Exception:
                continue
            if np_arr.size == 0:
                continue
            out_path = os.path.join(tr.outputDir, f"{prefix}_{reg}.npy")
            np.save(out_path, np_arr)

    print("All logs saved – job finished.")
    print("fastsim_cluster_main.py finished successfully.")


if __name__ == "__main__":
    main()
