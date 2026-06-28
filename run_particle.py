"""Single entrypoint: train all three region models for one particle type.

Usage:
    python run_particle.py <pdg> [--epochs N] [--output-dir DIR] [--dry-run]
"""

import argparse
import os
import time

import numpy as np

import utils
from pipeline.config import load_base, load_override, materialize_all
from pipeline.regions import REGIONS

LOG_MAP = {
    "KL_log": "KL", "D_wdist_log": "D", "Val_D_log": "ValD",
    "G_loss_log": "G", "Val_G_log": "ValG", "GP_log": "GP",
    "gradG_log": "GradG", "gradD_log": "GradD",
}


def build_configs(pdg, output_dir, num_epochs,
                  base_path="Config/base_cfg.json",
                  overrides_dir="Config/overrides"):
    base = load_base(base_path)
    override = load_override(overrides_dir, pdg)
    return materialize_all(base, override, pdg, output_dir, num_epochs)


def write_configs(configs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cfg in configs.values():
        utils.save_cfg(cfg)


def split_if_requested(master_csv, pdg, base_path, overrides_dir):
    import pandas as pd

    from pipeline.config import load_base, load_override
    from pipeline.data_split import split_particle

    base = load_base(base_path)
    override = load_override(overrides_dir, pdg)
    root = override.get("dataRoot", base["dataRoot"])
    prefix = override.get("dataPrefix", base.get("dataPrefix", ""))
    master = pd.read_csv(master_csv)
    return split_particle(master, pdg, out_dir=root, prefix=prefix)


def _run_training(configs, output_dir):
    # imported lazily so --dry-run needs no torch/data
    from trainerFactory import create_trainer
    from pipeline.calibration import fit_calibrator

    trainers = {r: create_trainer(cfg) for r, cfg in configs.items()}
    t0 = time.localtime()
    print(f"Start : {utils.get_time(t0)}")
    for region in REGIONS:
        print(f"--- Training {region} ---")
        tr = trainers[region]
        tr.run()
        gen_for_calib = tr.ema_gen if tr.ema_gen is not None else tr.genNet
        calib_path = os.path.join(output_dir, f"{region}_calib.pkl")
        fit_calibrator(gen_for_calib, tr.noiseDim, n_samples=200000, out_path=calib_path)
        print(f"{region} done : {utils.get_time(time.localtime(), t0)}")

    for region, tr in trainers.items():
        for attr, prefix in LOG_MAP.items():
            arr = getattr(tr, attr, None)
            if arr is not None and len(arr):
                np.save(os.path.join(tr.outputDir, f"{prefix}_{region}.npy"),
                        np.asarray(arr))
    print("All logs saved - job finished.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train fast-sim models for one particle.")
    parser.add_argument("pdg", type=int, help="PDG code of the particle")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base", type=str, default="Config/base_cfg.json")
    parser.add_argument("--overrides-dir", type=str, default="Config/overrides")
    parser.add_argument("--split-from", type=str, default=None,
                        help="Master CSV to split into per-region inputs before training.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Materialize and write configs only; no training.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir.endswith("/") else args.output_dir + "/"
    if args.split_from:
        paths = split_if_requested(args.split_from, args.pdg, args.base, args.overrides_dir)
        print(f"Split master into: {paths}")
    configs = build_configs(args.pdg, output_dir, args.epochs,
                            base_path=args.base, overrides_dir=args.overrides_dir)
    write_configs(configs, output_dir)
    if args.dry_run:
        print(f"Dry run: wrote configs for pdg {args.pdg} to {output_dir}")
        return
    _run_training(configs, output_dir)


if __name__ == "__main__":
    main()
