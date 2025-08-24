# fastsim_cluster_main.py – launcher (inner / outer1 / outer2), per-region lifecycle
# Changes integrated:
#  • Always save a "last" checkpoint per region (Gen/Critic) after training.
#  • If no "best" checkpoint exists (e.g., metric never crossed threshold), also save to the usual *_model.pt names.
#  • Print/record lightweight DataLoader summaries to help diagnose empty/too-small datasets.
#  • Persist a small per-region metadata JSON with basic run info.

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


def _fmt_time(t, t0=None):
    if t0 is None:
        return time.strftime("%Y-%m-%d %H:%M:%S", t)
    # format elapsed as HH:MM:SS
    dt = time.mktime(t) - time.mktime(t0)
    hrs = int(dt // 3600)
    mins = int((dt % 3600) // 60)
    secs = int(dt % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def _dl_summary(dl):
    """Return lightweight summary dict for a torch DataLoader (best-effort; avoids hard deps)."""
    summ = {}
    try:
        summ["batches"] = len(dl)
    except Exception:
        summ["batches"] = None
    summ["batch_size"] = getattr(dl, "batch_size", None)
    ds = getattr(dl, "dataset", None)
    try:
        summ["samples"] = len(ds) if ds is not None else None
    except Exception:
        summ["samples"] = None
    summ["drop_last"] = getattr(dl, "drop_last", None)
    return summ


def _save_region_artifacts(torch_mod, tr, reg, out_dir, valD_array):
    """
    Always save:
      • {reg}_Gen_last.pt
      • {reg}_Critic_last.pt
    And if the usual "best" files do not exist yet, also save:
      • {reg}_Gen_model.pt
      • {reg}_Critic_model.pt
    Additionally, write meta_{reg}.json with a few fields.
    """
    if torch_mod is None:
        print(f"[WARN] torch not available; skipping checkpoint save for {reg}.")
        return

    os.makedirs(out_dir, exist_ok=True)

    gen_last = os.path.join(out_dir, f"{reg}_Gen_last.pt")
    crit_last = os.path.join(out_dir, f"{reg}_Critic_last.pt")
    gen_best = os.path.join(out_dir, f"{reg}_Gen_model.pt")
    crit_best = os.path.join(out_dir, f"{reg}_Critic_model.pt")

    try:
        torch_mod.save(tr.genNet.state_dict(), gen_last)
        torch_mod.save(tr.critic.state_dict(), crit_last)
        print(f"[{reg}] Saved last checkpoints → {os.path.basename(gen_last)}, {os.path.basename(crit_last)}")
    except Exception as e:
        print(f"[WARN] Saving 'last' checkpoints failed for {reg}: {e}")

    # If "best" checkpoints don't exist (e.g., metric threshold never hit), create them from the current state.
    wrote_best_alias = False
    try:
        if not os.path.isfile(gen_best):
            torch_mod.save(tr.genNet.state_dict(), gen_best)
            wrote_best_alias = True
        if not os.path.isfile(crit_best):
            torch_mod.save(tr.critic.state_dict(), crit_best)
            wrote_best_alias = True
        if wrote_best_alias:
            print(f"[{reg}] Created best-alias checkpoints (no previous best).")
    except Exception as e:
        print(f"[WARN] Saving best-alias checkpoints failed for {reg}: {e}")

    # Meta JSON
    meta = {
        "region": reg,
        "dataGroup": getattr(tr, "dataGroup", None),
        "gen_last": os.path.basename(gen_last) if os.path.isfile(gen_last) else None,
        "critic_last": os.path.basename(crit_last) if os.path.isfile(crit_last) else None,
        "gen_best_exists": os.path.isfile(gen_best),
        "critic_best_exists": os.path.isfile(crit_best),
        "val_wdist_mean": float(np.mean(valD_array)) if valD_array is not None and len(valD_array) > 0 else None,
        "val_wdist_last": float(valD_array[-1]) if valD_array is not None and len(valD_array) > 0 else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(os.path.join(out_dir, f"meta_{reg}.json"), "w") as fp:
            json.dump(meta, fp, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write meta for {reg}: {e}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    regions = list(args.regions)

    t0 = time.localtime()
    print(f"Start     : {utils.get_time(t0)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Config dir: {args.cfg_dir}")
    print(f"Regions   : {regions}")

    # map Trainer attr -> filename prefix (kept to satisfy existing analysis in utils.check_run)
    log_map = {
        "KL_log"               : "KL",
        "Critic_wdist_log"     : "D",      # critic Wasserstein estimate (train; positive)
        "Val_Critic_wdist_log" : "ValD",   # critic Wasserstein estimate (val; positive)
        "G_loss_log"           : "G",
        "Val_G_loss_log"       : "ValG",
        "GP_log"               : "GP",
        "gradG_log"            : "GradG",
        "gradCritic_log"       : "GradD",
    }

    import gc
    try:
        import torch as _torch
    except Exception:
        _torch = None

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
        print(f"[{reg}] dataGroup = {getattr(tr, 'dataGroup', None)}")

        # Best-effort summaries for debugging empty DataLoaders / tiny datasets
        tr_dl_train = getattr(tr, "dl_train", None)
        tr_dl_val = getattr(tr, "dl_val", None)
        if tr_dl_train is not None:
            s_tr = _dl_summary(tr_dl_train)
            print(f"[{reg}] train DataLoader: {s_tr}")
        if tr_dl_val is not None:
            s_vl = _dl_summary(tr_dl_val)
            print(f"[{reg}]   val DataLoader: {s_vl}")

        print(f"--- Training {reg} ---")
        t_reg0 = time.localtime()
        tr.run()
        print(f"{reg} done : {_fmt_time(time.localtime(), t_reg0)} "
              f"(since start {_fmt_time(time.localtime(), t0)})")

        # persist this region's logs immediately
        for attr, prefix in log_map.items():
            arr = getattr(tr, attr, None)
            if arr is None:
                continue
            np_arr = np.asarray(arr)
            if np_arr.size == 0 or not np.all(np.isfinite(np_arr)):
                # Save finite subset where applicable; otherwise skip silently
                finite = np_arr[np.isfinite(np_arr)] if np_arr.size > 0 else np_arr
                if finite.size == 0:
                    continue
                out_path = os.path.join(tr.outputDir, f"{prefix}_{reg}.npy")
                np.save(out_path, finite)
                continue
            out_path = os.path.join(tr.outputDir, f"{prefix}_{reg}.npy")
            np.save(out_path, np_arr)

        # ALWAYS save a "last" checkpoint for this region
        # If best files don't exist yet, create them too (alias the last).
        valD_arr = None
        try:
            valD_arr = np.asarray(getattr(tr, "Val_Critic_wdist_log", []))
        except Exception:
            pass
        _save_region_artifacts(_torch, tr, reg, args.output_dir, valD_arr)

        # free memory before next region
        del tr
        gc.collect()
        if _torch is not None:
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass

    print("All logs saved – job finished.")
    print("fastsim_cluster_main.py finished successfully.")


if __name__ == "__main__":
    main()
