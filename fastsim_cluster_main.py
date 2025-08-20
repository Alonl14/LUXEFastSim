# fastsim_cluster_main.py – launcher (inner / outer1 / outer2), per-region lifecycle
import os
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
    p.add_argument("config_stamp", type=str, help='Config suffix (e.g., "_cfg_v7.json" or "_cfg_cluster.json").')
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
    regions = list(args.regions)

    t0 = time.localtime()
    print(f"Start : {utils.get_time(t0)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Config dir: {args.cfg_dir}")
    print(f"Regions   : {regions}")

    # map Trainer attr -> filename prefix (kept to satisfy existing analysis in utils.check_run)
    log_map = {
        "KL_log"               : "KL",
        "Critic_wdist_log"     : "D",      # critic Wasserstein estimate (train)
        "Val_Critic_wdist_log" : "ValD",   # critic Wasserstein estimate (val)
        "G_loss_log"           : "G",
        "Val_G_loss_log"       : "ValG",
        "GP_log"               : "GP",
        "gradG_log"            : "GradG",
        "gradCritic_log"       : "GradD",
    }

    import gc
    try:
        import torch
    except Exception:
        torch = None

    for reg in regions:
        cfg_path = os.path.join(args.cfg_dir, f"{reg}{args.config_stamp}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing config for region '{reg}': {cfg_path}")

        with open(cfg_path, "r") as fp:
            cfg = json.load(fp)

        # patch runtime fields
        cfg.update({
            "outputDir": args.output_dir,
            "numEpochs": args.num_epochs,
            "seed": args.seed,
            # training-time memory: don't keep large pre-QT copies
            "keepPreprocess": False,
        })

        print(f"\n--- Building trainer for {reg} ---")
        tr = create_trainer(cfg)

        print(f"--- Training {reg} ---")
        t_reg0 = time.localtime()
        tr.run()
        print(f"{reg} done : {utils.get_time(time.localtime(), t_reg0)} "
              f"(since start {utils.get_time(time.localtime(), t0)})")

        # persist this region's logs immediately
        for attr, prefix in log_map.items():
            arr = getattr(tr, attr, None)
            if arr is None:
                continue
            np_arr = np.asarray(arr)
            if np_arr.size == 0:
                continue
            out_path = os.path.join(tr.outputDir, f"{prefix}_{reg}.npy")
            np.save(out_path, np_arr)

        # free memory before next region
        del tr
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print("All logs saved – job finished.")
    print("fastsim_cluster_main.py finished successfully.")


if __name__ == "__main__":
    main()
