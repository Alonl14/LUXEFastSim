# trainerFactory.py – builds per-region, cfg-driven Generator/Critic + loaders + optim
import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import utils
from dataset import ParticleDataset
from generator import Generator
from critic import Critic


def _select_device(cfg_device=None) -> torch.device:
    if cfg_device:
        want = str(cfg_device).lower()
        if want.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            print("[trainerFactory] CUDA requested but not available → falling back to CPU.")
            return torch.device("cpu")
        if want == "cpu":
            return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _smart_load_state_dict(model: nn.Module, path: str, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)


def _suggest_layers(n: int, kind: str):
    # size-aware defaults; kind in {"gen","crit"}
    if n >= 10_000_000:
        return [512, 1024, 1024, 512] if kind == "gen" else [512, 512, 512, 512]
    if n >= 1_000_000:
        return [512, 512, 512] if kind == "gen" else [512, 512, 512]
    if n >= 100_000:
        return [256, 256, 256] if kind == "gen" else [256, 256, 256]
    return [128, 128] if kind == "gen" else [128, 128]


def create_trainer(cfg: Dict[str, Any], trained: bool = False):
    t0 = time.localtime()
    print(f"[trainerFactory] Creating trainer at: {utils.get_time(t0)}")

    # -------------------- IO & cfg --------------------
    output_dir = cfg.get("outputDir") or os.path.join(os.getcwd(), "Output")
    os.makedirs(output_dir, exist_ok=True)
    cfg["outputDir"] = output_dir
    utils.save_cfg(cfg)

    # -------------------- dataset --------------------
    print("[trainerFactory] Loading dataset…")
    ds = ParticleDataset(cfg)
    n_samples = len(ds)
    print(f"[trainerFactory] Dataset created: {n_samples} rows")

    val_frac = float(cfg.get("valFrac", 0.20))
    val_size = int(round(val_frac * n_samples))
    train_size = n_samples - val_size
    seed = int(cfg.get("seed", 1337))
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)
    print(f"[trainerFactory] Split train/val = {train_size}/{val_size} (seed={seed})")

    # -------------------- device + loaders ------------
    device = _select_device(cfg.get("device"))
    use_cuda = device.type == "cuda"
    print(f"[trainerFactory] Using device: {device}")

    num_workers = int(cfg.get("numWorkers", min(8, os.cpu_count() or 1)))
    pin_memory = bool(cfg.get("pinMemory", use_cuda))
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batchSize"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batchSize"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    print("[trainerFactory] DataLoaders ready.")

    # -------------------- models ----------------------
    num_features = len(cfg["features"].keys())

    gen_layers  = cfg.get("genLayers")    or _suggest_layers(n_samples, kind="gen")
    crit_layers = cfg.get("criticLayers") or _suggest_layers(n_samples, kind="crit")

    gen_norm = cfg.get("genNorm", "layer")
    gen_act = cfg.get("genAct", "relu")
    gen_dropout = float(cfg.get("genDropout", 0.0))

    crit_norm = cfg.get("criticNorm", "layer")
    crit_act = cfg.get("criticAct", "lrelu")
    crit_dropout = float(cfg.get("criticDropout", 0.0))

    gen = Generator(
        noiseDim=int(cfg["noiseDim"]),
        numFeatures=num_features,
        hidden_dims=gen_layers,
        norm=gen_norm,
        activation=gen_act,
        bias_last=True,
        bias_hidden=False,
        dropout=gen_dropout,
    ).to(device)

    critic = Critic(
        numFeatures=num_features,
        hidden_dims=crit_layers,
        norm=crit_norm,
        activation=crit_act,
        bias_hidden=True,
        dropout=crit_dropout,
        out_bias=False,
    ).to(device)

    # Optional checkpoints
    gen_ckpt   = cfg.get("genStateDict", None)
    # backward-compat: accept old key name too
    crit_ckpt  = cfg.get("criticStateDict", cfg.get("discStateDict", None))

    if gen_ckpt:
        print(f"[trainerFactory] Loading generator weights: {gen_ckpt}")
        _smart_load_state_dict(gen, gen_ckpt, device)
    if crit_ckpt:
        print(f"[trainerFactory] Loading critic weights: {crit_ckpt}")
        _smart_load_state_dict(critic, crit_ckpt, device)

    # -------------------- optimizers ------------------
    gen_lr  = float(cfg.get("generatorLearningRate", 1.5e-5))
    crit_lr = float(cfg.get("criticLearningRate",    1.0e-6))
    betas   = tuple(cfg.get("betas", (0.5, 0.9)))

    optG = optim.Adam(gen.parameters(),    lr=gen_lr,  betas=betas)
    optD = optim.Adam(critic.parameters(), lr=crit_lr, betas=betas)

    # -------------------- pack for Trainer ------------
    trainer_cfg = {
        "genNet": gen,
        "criticNet": critic,            # NEW key
        "genOptimizer": optG,
        "criticOptimizer": optD,        # NEW key
        "noiseDim": int(cfg["noiseDim"]),
        "outputDir": output_dir,
        "dataloader": train_loader,
        "valDataloader": val_loader,
        "dataset": ds,
        "dataGroup": cfg["dataGroup"],
        "numEpochs": int(cfg["numEpochs"]),
        "device": device,
        "nCrit": int(cfg["nCrit"]),
        "Lambda": float(cfg["Lambda"]),
        "GMaxSteps": cfg.get("GMaxSteps", None),
        "gradMetric": cfg.get("gradMetric", "norm"),
        "genStateDict": gen_ckpt,
        "criticStateDict": crit_ckpt,
        "arch": {
            "genLayers": gen_layers,
            "criticLayers": crit_layers,
            "genAct": gen_act, "genNorm": gen_norm,
            "criticAct": crit_act, "criticNorm": crit_norm,
        }
    }

    from trainer import Trainer
    print("[trainerFactory] Building Trainer…")
    tr = Trainer(cfg=trainer_cfg)
    print(f"[trainerFactory] Trainer ready in {utils.get_time(time.localtime(), t0)}")
    return tr


class TrainerFactory:
    """Deprecated: kept for compatibility; use create_trainer(cfg) instead."""
    pass
