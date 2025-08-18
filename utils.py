# utils.py
import os
import gc
import json
import time
import shutil
import random
import subprocess
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import kstest as ks
from scipy.stats import binned_statistic_dd
from numba import njit
import psutil
import tqdm

import dataset
from generator import Generator

# ---- Matplotlib global style (was change 5) ---------------------------------
plt.style.use('seaborn-v0_8-deep')
plt.rcParams.update({'font.size': 25})


# ============================================================================ #
# Basic helpers                                                                #
# ============================================================================ #

def get_time(now: Optional[time.struct_time] = None,
             start: Optional[time.struct_time] = None) -> str:
    """
    If only `now` is provided (or both None), returns HH:MM:SS of local time.
    If both `now` and `start` provided, returns elapsed as HH:MM:SS.
    Backwards compatible with previous calls like get_time(beg_time) and get_time(end, beg).
    """
    if now is None:
        now = time.localtime()
    if start is None:
        return time.strftime("%H:%M:%S", now)
    # real elapsed wall time
    t_now = time.mktime(now)
    t_start = time.mktime(start)
    delta = max(0, int(round(t_now - t_start)))
    hh = delta // 3600
    mm = (delta % 3600) // 60
    ss = delta % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def save_cfg(cfg: Dict[str, Any]) -> None:
    """Persist cfg to outputDir/cfg_<dataGroup>.json"""
    out_dir = cfg.get('outputDir') or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cfg_{cfg['dataGroup']}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def weights_init(m: nn.Module) -> None:
    """He init for Linear; safe guards for norm layers (affine only)."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d)):
        # Only touch weights if affine exists
        if hasattr(m, "affine") and m.affine and hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================================ #
# Statistics / distances                                                       #
# ============================================================================ #

def compute_sparse_histogram(data: torch.Tensor, bin_edges):
    hist, _, _2 = binned_statistic_dd(
        data.detach().cpu().numpy(),
        values=None, statistic='count', bins=bin_edges
    )
    print(f"Size of hist in bytes: {hist.nbytes}")
    return torch.tensor(hist, dtype=torch.float32)


def get_kld(real: torch.Tensor,
            fake: torch.Tensor,
            bins: int = 12,
            epsilon: float = 1e-12) -> torch.Tensor:
    """
    KL divergence between real and fake using shared bin edges (from real).
    """
    real_np = real.detach().cpu().numpy()
    fake_np = fake.detach().cpu().numpy()

    n_features = real_np.shape[1]
    if isinstance(bins, int):
        bin_counts = [bins] * n_features
    else:
        assert len(bins) == n_features, "bins must be int or sequence matching features"
        bin_counts = bins

    bin_edges = [np.linspace(real_np[:, i].min(), real_np[:, i].max(), b + 1)
                 for i, b in enumerate(bin_counts)]

    real_hist, _ = np.histogramdd(real_np, bins=bin_edges, density=True)
    fake_hist, _ = np.histogramdd(fake_np, bins=bin_edges, density=True)

    real_p = real_hist.flatten()
    fake_p = fake_hist.flatten()

    real_p = real_p / (real_p.sum() + epsilon)
    fake_p = fake_p / (fake_p.sum() + epsilon)

    real_p = real_p + epsilon
    fake_p = fake_p + epsilon

    real_t = torch.from_numpy(real_p).float()
    fake_t = torch.from_numpy(fake_p).float()

    return torch.sum(real_t * (real_t.log() - fake_t.log()))


# ============================================================================ #
# Physics feature utilities                                                    #
# ============================================================================ #

class Particle:
    def __init__(self, name, mass, charge, spin, isospin, pdg):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.isospin = isospin
        self.pdg = pdg

    def __repr__(self):
        return (f"Particle(name='{self.name}', mass={self.mass}, charge={self.charge}, "
                f"spin={self.spin}, isospin={self.isospin}, pdg={self.pdg})\n")

    def at(self, features):
        return [getattr(self, feature) for feature in features]


pdg_dict = {
    11: ["Electron", 0.0005109989461, -1, 0.5, 0],
    -11: ["Positron", 0.0005109989461, +1, 0.5, 0],
    12: ["Neutrino", 0.0, 0, 0.5, 0],
    -12: ["Anti-Neutrino", 0.0, 0, 0.5, 0],
    13: ["Muon", 0.1056583745, -1, 0.5, 0],
    -13: ["Anti-Muon", 0.1056583745, +1, 0.5, 0],
    14: ["Muon-Neutrino", 0.0, 0, 0.5, 0],
    -14: ["Anti-Muon-Neutrino", 0.0, 0, 0.5, 0],
    22: ["Photon", 0.0, 0, 1, 0],
    130: ["K0-L", 0.497611, 0, 0, 0.5],
    211: ["Pion+", 0.13957039, +1, 0, 0.5],
    -211: ["Pion-", 0.13957039, -1, 0, 0.5],
    2112: ["Neutron", 0.93956, 0, 0.5, 0.5],
    2212: ["Proton", 0.93827, +1, 0.5, 0.5],
    1000010020: ["Deuteron", 1.875612573, +1, 1, 0],
    1000020040: ["Alpha", 3.727379, +2, 0, 0],
    1000030070: ["Li-7", 6.533839, +3, 1.5, 1.5],
    1000060120: ["C-12", 11.177930, +6, 0, 0],
    1000080160: ["O-16", 14.899170, +8, 0, 0],
    1000260540: ["Fe-54", 50.832600, +26, 0, 0]
}
particle_dict = {
    pdg: Particle(name, mass, charge, spin, isospin, pdg)
    for pdg, (name, mass, charge, spin, isospin) in pdg_dict.items()
}


def add_features(df: pd.DataFrame, pdg: int) -> None:
    """
    Adds redundant features so that all possible features are present.
    Expects columns like [' xx',' yy',' pxx',' pyy',' pzz',' eneg',' time'] present.
    """
    mass = particle_dict[pdg].mass
    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)

    if ' pxx' in df.columns and ' pyy' in df.columns:
        df[' rp'] = np.sqrt(df[' pxx'] ** 2 + df[' pyy'] ** 2)
        df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx']) + np.pi
    else:
        df[' pxx'] = df[' rp'] * np.cos(df[' phi_p'] - np.pi)
        df[' pyy'] = df[' rp'] * np.sin(df[' phi_p'] - np.pi)

    # Kinematic identity for energy (kept as in your original, but safer numerics)
    # eneg = (rp^2 + pz^2) / (sqrt(rp^2 + pz^2 + m^2) + m)
    rp2 = (df[' rp'] ** 2)
    pz2 = (df[' pzz'] ** 2)
    denom = np.sqrt(rp2 + pz2 + mass ** 2) + mass + 1e-12
    df[' eneg'] = (rp2 + pz2) / denom

    # theta via atan2 for stability
    df['theta'] = np.arctan2(df[' rp'], df[' pzz'])


def make_polar_features(df: pd.DataFrame) -> None:
    df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)
    df[' rp'] = np.sqrt(df[' pxx'] ** 2 + df[' pyy'] ** 2)
    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx']) + np.pi
    df['theta'] = np.arccos(df[' pzz'] / np.sqrt(df[' pzz'] ** 2 + df[' rp'] ** 2 + 1e-12))


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into inner / outer1 / outer2 by geometry cuts."""
    outer1_pos = (((df[' xx'] >= 500) | (df[' yy'] >= 500)) |
                  ((df[' yy'] >= 400) & (df[' xx'] <= -1700)))
    inner_pos = ~outer1_pos & (df[' xx'] >= -1700)
    outer2_pos = (df[' yy'] <= 600) & (df[' xx'] <= -1700)

    inner_df = df[inner_pos]
    outer1_df = df[outer1_pos]
    outer2_df = df[outer2_pos]
    print(f'Created 3 datasets:'
          f'\nPoints in region I: {len(outer1_df)}'
          f'\nPoints in region II: {len(outer2_df)}'
          f'\nPoints in region III: {len(inner_df)}')
    return inner_df, outer1_df, outer2_df


def get_q(ds) -> np.ndarray:
    """Get quantile array from dataset."""
    return ds.quantiles.quantiles_


# ============================================================================ #
# Plotting                                                                     #
# ============================================================================ #

def plot_correlations(x, y, xlabel, ylabel, run_id, key,
                      bins=[400, 400], loglog=False, Xlim=None, Ylim=None, path=None):
    H, xb, yb = np.histogram2d(
        x, y, bins=bins,
        range=[[x.min(), x.max()], [y.min(), y.max()]],
        density=True
    )
    X, Y = np.meshgrid(xb, yb)
    plt.figure(dpi=200)

    # defensively avoid zero minima for LogNorm
    eps = 1e-12
    if xlabel == "x[mm]":
        vmin, vmax = 1e-10, 1e-6
    elif xlabel == "t[ns]":
        vmin, vmax = 1e-6, 1e3
    elif xlabel == r'\phi_p [rad]':
        vmin, vmax = 1, 1e-4
    else:
        vmin, vmax = 1e-6, 1e-3
    vmin = max(vmin, eps)

    plt.pcolormesh(X, Y, H.T, norm=LogNorm(vmin=vmin, vmax=vmax))
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if Xlim is not None:
        plt.xlim(Xlim)
    if Ylim is not None:
        plt.ylim(Ylim)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(r"$" + xlabel + r"$", fontsize=25)
    plt.ylabel(r"$" + ylabel + r"$", fontsize=25)
    plt.grid(True)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    plt.tight_layout()

    if run_id is not None:
        hist_path = os.path.join(path, 'plots', '2dHists')
        key_dir = os.path.join(hist_path, key)
        os.makedirs(key_dir, exist_ok=True)
        clean_xlabel = ''.join(ch for ch in xlabel if ch.isalnum())
        clean_ylabel = ''.join(ch for ch in ylabel if ch.isalnum())
        out = os.path.join(key_dir, f"2d-{clean_xlabel}-{clean_ylabel}.png")
        plt.savefig(out, bbox_inches='tight')
        plt.close()
    else:
        hist_path = "/storage/agrp/alonle/GAN_Output"
        os.makedirs(hist_path, exist_ok=True)
        out = os.path.join(hist_path, f"{key}{xlabel}-{ylabel}.png")
        plt.savefig(out)
        plt.close()
    return H


def make_plots(df, dataGroup, run_id="", key="", path=""):
    # region-dependent limits (kept from your original logic)
    x_lim = [-4000, 4000]
    y_lim = [-4000, 4000]
    if dataGroup == 'inner':
        x_lim = [-1700, 500]
        y_lim = [-2500, 500]

    Hxy = plot_correlations(df[' xx'], df[' yy'], 'x[mm]', 'y[mm]', run_id, key, path=path)
    energy_bins = 10 ** np.linspace(-13, 0, 400)
    bin_stop = np.log10(np.max(df[' time']))
    time_bins = 10 ** np.linspace(1, bin_stop + 0.5, 400)
    Het = plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', run_id, key,
                            bins=[time_bins, energy_bins], loglog=True, path=path)
    Hrth = plot_correlations(df[' rx'], df['theta'], 'r [mm]', '\\theta_p [rad]', run_id, key, path=path)
    Hpp = plot_correlations(df[' phi_p'], df[' phi_x'], '\phi_p [rad]', '\phi_x [rad]', run_id, key, path=path)
    return Hxy, Het, Hrth, Hpp


def plot_1d(data, DF, feat, ks_placeholder, fig_path, key):
    plt.figure(dpi=200)
    plt.yscale('log')
    bins = np.linspace(np.min(DF[feat]), np.max(DF[feat]), 200)
    if feat == ' time' or feat == ' eneg':
        # guard for positives
        mn = max(np.min(np.abs(DF[feat])), 1e-12)
        mx = max(np.sort(DF[feat])[-10], mn * 10)
        bins = np.logspace(np.log10(mn), np.log10(mx), 200)
        plt.xscale('log')

    plt.hist(DF[feat], bins=bins, density=True, alpha=0.6)
    plt.hist(data[feat], bins=bins, density=True, alpha=0.6)
    plt.legend(["Generated data", "FullSim data"])
    labeldict = {' xx': '$x~$[mm]',
                 ' yy': '$y~$[mm]',
                 ' pxx': '$p_x~$[GeV]',
                 ' pyy': '$p_y~$[GeV]',
                 ' pzz': '$p_z~$[GeV]',
                 ' eneg': '$E~$[GeV]',
                 ' time': '$t~$[ns]',
                 ' rx': '$r_x~$[mm]',
                 ' phi_x': '$\\phi_x~$[rad]',
                 ' rp': '$r_p~$[GeV]',
                 ' phi_p': '$\\phi_p~$[rad]',
                 'theta': '$\\theta~$[rad]'}
    plt.xlabel(labeldict.get(feat, feat))
    out_dir = os.path.join(fig_path, '1dHists', key)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{feat.strip().capitalize()}.png")
    plt.savefig(out, bbox_inches='tight')
    plt.close()


# ============================================================================ #
# Data generation & analysis                                                   #
# ============================================================================ #

def generate_ds(generator_net: nn.Module, factor: int, cfg: Dict[str, Any]):
    """
    Generate fake/real dataframes using a trained generator and dataset’s inverse QT.
    """
    print(f"Starting to work on dataset for region {cfg['dataGroup']}...")
    ds = dataset.ParticleDataset(cfg)
    print(f"Created dataset for region {cfg['dataGroup']} with {len(ds.data)} events")
    numEvents = int(ds.data.shape[0])
    generator_net.eval().to('cpu')
    n = max(1, numEvents // max(1, int(factor)))
    with torch.no_grad():
        generated = generator_net(torch.randn(n, cfg['noiseDim'], device='cpu')).detach().cpu().numpy()

    features = list(cfg['features'].keys())
    ds.data = pd.DataFrame(np.empty((0, len(features))), columns=features)
    data_values = ds.quantiles.inverse_transform(generated) if ds.quantiles is not None else generated
    for i, feature in enumerate(features):
        ds.data[feature] = data_values[:, i]
    del data_values
    ds.apply_transformation(cfg, inverse=True)
    fake, real = ds.data, ds.preprocess
    del ds
    gc.collect()
    return fake, real


def generate_fake_real_dfs(run_id: str, cfg: Dict[str, Any], run_dir: str, generator_net: Optional[nn.Module] = None):
    """
    Load trained generator from run_dir, build it with the SAME architecture specified in cfg,
    then produce fake & real DataFrames, and add derived features.
    """
    numFeatures = len(cfg["features"].keys())
    model_path = os.path.join(run_dir, f"{cfg['dataGroup']}_Gen_model.pt")
    dim = int(cfg["noiseDim"])

    # Arch + hyperparams from cfg (fallbacks mirror training defaults)
    gen_layers = cfg.get("genLayers", [512, 1024, 1024, 512])
    gen_norm = cfg.get("genNorm", "layer")
    gen_act = cfg.get("genAct", "relu")
    gen_dropout = float(cfg.get("genDropout", 0.0))

    if generator_net is None:
        generator_net = Generator(
            noiseDim=dim,
            numFeatures=numFeatures,
            hidden_dims=gen_layers,
            norm=gen_norm,
            activation=gen_act,
            bias_last=True,
            bias_hidden=False,
            dropout=gen_dropout,
        )
        state = torch.load(model_path, map_location=torch.device('cpu'))
        try:
            generator_net.load_state_dict(state)
        except RuntimeError:
            # strip possible 'module.' prefixes (old DataParallel)
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            generator_net.load_state_dict(new_state, strict=False)

    fake_df, real_df = generate_ds(generator_net, factor=1, cfg=cfg)

    # time / space filters (kept from your original)
    fake_df = fake_df[fake_df[' time'] <= 1e6]
    real_df = real_df[real_df[' time'] <= 1e6]

    add_features(fake_df, cfg['pdg'])
    add_features(real_df, cfg['pdg'])
    return fake_df, real_df



def is_latex_available() -> bool:
    latex = shutil.which("latex")
    dvipng = shutil.which("dvipng")
    gs = shutil.which("gs")
    return latex is not None and (dvipng is not None or gs is not None)


def compact_latex(n: float) -> str:
    coeff, exp = "{:.2e}".format(n).split('e')
    return f"{coeff} \\cdot 10^{{{int(exp)}}}"


def make_ed_fig(null, H1, group, fig_path, real_tag="Y", fake_tag="X", plotting=True):
    ks_test = ks(null, H1)
    pval, ks_stat = ks_test.pvalue, ks_test.statistic
    if plotting:
        fig, axs = plt.subplots(dpi=200)
        if is_latex_available():
            try:
                subprocess.run(["latex", "-version"], check=True, stdout=subprocess.DEVNULL)
                plt.rcParams['text.usetex'] = True
                print("LaTeX rendering enabled.")
            except Exception as e:
                print("LaTeX found, but couldn't be used:", e)
        else:
            print("LaTeX or required tools not found; using default matplotlib text.")
        axs.grid(True, which='both', color='0.65', linestyle='-')
        bin_max = np.max(np.concatenate((null, H1)))
        bin_min = np.min(np.concatenate((null, H1)))
        bins = np.linspace(bin_min, bin_max, 50)
        axs.hist(null, bins=bins, density=True, alpha=0.6)
        axs.hist(H1, bins=bins, density=True, alpha=0.6)
        if pval < 0.1:
            axs.text(.98, .5, f' $p$-value: ${compact_latex(pval)}$ ', ha='right', va='top', transform=axs.transAxes)
            axs.text(.98, .4, f' $K_{{n,m}}={ks_stat:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
        else:
            axs.text(.98, .5, f' $p$-value: ${pval:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
            axs.text(.98, .4, f' $K_{{n,m}}={ks_stat:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
        plt.xlabel("$D_E \\rm{[a.u]}$")
        plt.ylabel("frequency")
        plt.yscale('log')
        axs.legend([f"$D_E({real_tag},{fake_tag})$", f"$D_E({real_tag},{real_tag}^\\prime)$"])
        os.makedirs(fig_path, exist_ok=True)
        out = os.path.join(fig_path, f"{group}_histograms.png")
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
    return pval, ks_stat


# ============================================================================ #
# Energy distance utilities                                                    #
# ============================================================================ #

@njit
def euclidean_distance_matrix(x, y, out):
    n_x, dim = x.shape
    n_y = y.shape[0]
    for i in range(n_x):
        for j in range(n_y):
            s = 0.0
            for k in range(dim):
                s += (x[i, k] - y[j, k]) ** 2
            out[i, j] = np.sqrt(s)
    return out


def get_batches(array, batch_size):
    batch_list = []
    i = 0
    while i + batch_size <= array.shape[0]:
        batch_list.append(array[i:i + batch_size, :])
        i += batch_size
    return batch_list


def get_ed(x, y):
    n_x = x.shape[0]
    n_y = y.shape[0]
    buf = np.empty((x.shape[0], x.shape[0]), dtype=np.float32)
    D_XX = euclidean_distance_matrix(x, x, out=buf)
    XX_mean = np.sum(D_XX) / (n_x ** 2)
    del D_XX
    buf = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
    D_XY = euclidean_distance_matrix(x, y, out=buf)
    XY_mean = np.sum(D_XY) / (n_x * n_y)
    del D_XY
    buf = np.empty((y.shape[0], y.shape[0]), dtype=np.float32)
    D_YY = euclidean_distance_matrix(y, y, out=buf)
    YY_mean = np.sum(D_YY) / (n_y ** 2)
    del D_YY
    del buf
    return 2 * XY_mean - XX_mean - YY_mean


def get_batch_ed_histograms(x_c: pd.DataFrame, y_c: pd.DataFrame, batch_size=1000):
    """
    Normalize features (z-score) per column; return ED(null) and ED(H1) over batches.
    """
    x = x_c.copy()
    y = y_c.copy()

    for f in x.columns:
        x[f] = (x[f] - np.mean(x[f])) / (np.std(x[f]) + 1e-12)
        y[f] = (y[f] - np.mean(y[f])) / (np.std(y[f]) + 1e-12)

    x_batches = get_batches(x.values, batch_size)
    y_batches = get_batches(y.values, batch_size)

    y_prime_batches = y_batches.copy()
    random.shuffle(y_prime_batches)

    n_batches = min(len(x_batches), len(y_batches))
    ED_null = np.zeros(n_batches)
    ED_H1 = np.zeros(n_batches)
    for i in tqdm.tqdm(range(n_batches)):
        x_batch, y_batch, y_prime_batch = (x_batches[i], y_batches[i], y_prime_batches[i])
        ED_null[i] = get_ed(x_batch, y_batch)
        ED_H1[i] = get_ed(y_prime_batch, y_batch)
    return ED_null, ED_H1


def get_sampled_ed_histograms(x: pd.DataFrame, y: pd.DataFrame,
                              batch_size=1000, num_iters=10000):
    """
    Sampled version: draws random batches with replacement for a fixed number of iterations.
    """
    x_processed = x.copy()
    y_processed = y.copy()

    for f in x.columns:
        x_processed[f] = (x_processed[f] - np.mean(x_processed[f])) / (np.std(x_processed[f]) + 1e-12)
        y_processed[f] = (y_processed[f] - np.mean(y_processed[f])) / (np.std(y_processed[f]) + 1e-12)

    x_indices = np.random.choice(len(x_processed), size=(num_iters, batch_size), replace=True)
    y_indices = np.random.choice(len(y_processed), size=(num_iters, batch_size), replace=True)
    y_prime_indices = np.random.choice(len(y_processed), size=(num_iters, batch_size), replace=True)

    x_values = x_processed.values
    y_values = y_processed.values

    ED_null = np.zeros(num_iters)
    ED_H1 = np.zeros(num_iters)

    for i in tqdm.tqdm(range(num_iters), desc='Calculating ED', leave=False):
        x_batch = x_values[x_indices[i]]
        y_batch = y_values[y_indices[i]]
        y_prime_batch = y_values[y_prime_indices[i]]

        ED_null[i] = get_ed(x_batch, y_batch)
        ED_H1[i] = get_ed(y_prime_batch, y_batch)

    return ED_null, ED_H1


def get_distance(data, DF, feat):
    bins = np.linspace(np.min(data[feat]), np.max(data[feat]), 1000)
    if feat in (' time', ' eneg'):
        bins = np.linspace(np.min(data[feat]), np.sort(data[feat])[-10], 1000)
    h1 = np.histogram(data[feat], bins=bins, density=True)[0]
    h2 = np.histogram(DF[feat], bins=bins, density=True)[0]
    mean = 0.0
    for i in range(len(h1)):
        if h1[i] == 0 and h2[i] == 0:
            mean += 0
        else:
            mean += np.abs(h2[i] - h1[i]) / (h1[i] + h2[i] + 1e-12)
    return mean / max(1, len(h1))


# ============================================================================ #
# Run analysis / plotting driver                                               #
# ============================================================================ #

def fix_path(cfg: Dict[str, Any], feature: str) -> None:
    """
    TEMPORARY: fix CSV paths when switching cluster→local by replacing with TrainData/<file>.
    """
    file_name = os.path.basename(cfg[feature])
    cfg[feature] = os.path.join("TrainData", file_name)


def check_run(run_id: str,
              path: Optional[str] = None,
              calculate_BED: bool = True,
              save_df: bool = False,
              plot_metrics: bool = True,
              plot_results: bool = True):
    plt.ioff()
    if path is None:
        print("Local path")
        run_dir = os.path.join('Output', f'run_{run_id}')
    else:
        print("Cluster path")
        run_dir = os.path.join(path, f'run_{run_id}')

    fig_path = os.path.join(run_dir, 'plots')
    os.makedirs(fig_path, exist_ok=True)

    with open(os.path.join(run_dir, "cfg_inner.json"), 'r') as f:
        cfg_inner = json.load(f)
    with open(os.path.join(run_dir, "cfg_outer1.json"), 'r') as f:
        cfg_outer1 = json.load(f)
    with open(os.path.join(run_dir, "cfg_outer2.json"), 'r') as f:
        cfg_outer2 = json.load(f)

    if path is None:
        fix_path(cfg_inner, "data_path")
        fix_path(cfg_outer1, "data_path")
        fix_path(cfg_outer2, "data_path")

    # Generate dataframes
    if plot_results or save_df or calculate_BED:
        generation_time_a = time.localtime()

        innerDF, innerData = generate_fake_real_dfs(run_id, cfg_inner, run_dir)
        posIn = (innerDF[' time'] <= 1e6) & (innerDF[' rx'] <= 4000) & (innerDF[' xx'] <= 500) & \
                (innerDF[' xx'] >= -1700) & (innerDF[' yy'] <= 520)
        innerDF = innerDF[posIn]
        print(f"[mem after inner init] {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        outer1DF, outer1Data = generate_fake_real_dfs(run_id, cfg_outer1, run_dir)
        posOut1 = (outer1DF[' time'] <= 1e6) & (outer1DF[' rx'] <= 4000) & \
                  ((outer1DF[' xx'] >= 500) | (outer1DF[' yy'] >= 520))
        outer1DF = outer1DF[posOut1]
        print(f"[mem after outer1 init] {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        outer2DF, outer2Data = generate_fake_real_dfs(run_id, cfg_outer2, run_dir)
        posOut2 = (outer2DF[' time'] <= 1e6) & (outer2DF[' rx'] <= 4000) & \
                  ((outer2DF[' xx'] < -1700) & (outer2DF[' yy'] <= 520))
        outer2DF = outer2DF[posOut2]
        print(f"[mem after outer2 init] {psutil.Process().memory_info().rss / 1e9:.2f} GB")

        generation_time_b = time.localtime()
        print(f'Created DFs in {get_time(generation_time_b, generation_time_a)}')

    # BED
    if calculate_BED:
        print("getting batch ED...")
        batch_size = 500
        small_batch = 50
        inner_null_values, inner_H1_values = get_batch_ed_histograms(innerDF, innerData, batch_size=batch_size)
        outer1_null_values, outer1_H1_values = get_batch_ed_histograms(outer1DF, outer1Data, batch_size=batch_size)
        outer2_null_values, outer2_H1_values = get_batch_ed_histograms(outer2DF, outer2Data, batch_size=small_batch)
        make_ed_fig(inner_null_values, inner_H1_values, 'inner', fig_path)
        make_ed_fig(outer1_null_values, outer1_H1_values, 'outer1', fig_path)
        make_ed_fig(outer2_null_values, outer2_H1_values, 'outer2', fig_path)
        print(f"[mem after ED] {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    # Metrics plots
    if plot_metrics:
        # keep filenames for backward-compatibility w/ main launcher
        iKLPath   = os.path.join(run_dir, "KL_inner.npy")
        oKL1Path  = os.path.join(run_dir, "KL_outer1.npy")
        oKL2Path  = os.path.join(run_dir, "KL_outer2.npy")

        iDPath    = os.path.join(run_dir, "D_inner.npy")
        oD1Path   = os.path.join(run_dir, "D_outer1.npy")
        oD2Path   = os.path.join(run_dir, "D_outer2.npy")

        iVDPath   = os.path.join(run_dir, "ValD_inner.npy")
        oVD1Path  = os.path.join(run_dir, "ValD_outer1.npy")
        oVD2Path  = os.path.join(run_dir, "ValD_outer2.npy")

        iGPath    = os.path.join(run_dir, "G_inner.npy")
        oG1Path   = os.path.join(run_dir, "G_outer1.npy")
        oG2Path   = os.path.join(run_dir, "G_outer2.npy")

        iVGPath   = os.path.join(run_dir, "ValG_inner.npy")
        oVG1Path  = os.path.join(run_dir, "ValG_outer1.npy")
        oVG2Path  = os.path.join(run_dir, "ValG_outer2.npy")

        iGradGPath  = os.path.join(run_dir, "GradG_inner.npy")
        oGradG1Path = os.path.join(run_dir, "GradG_outer1.npy")
        oGradG2Path = os.path.join(run_dir, "GradG_outer2.npy")

        iGradDPath  = os.path.join(run_dir, "GradD_inner.npy")
        oGradD1Path = os.path.join(run_dir, "GradD_outer1.npy")
        oGradD2Path = os.path.join(run_dir, "GradD_outer2.npy")

        # load
        innerKLDiv  = np.load(iKLPath)
        outer1KLDiv = np.load(oKL1Path)
        outer2KLDiv = np.load(oKL2Path)

        innerWdist, outer1Wdist, outer2Wdist = np.load(iDPath), np.load(oD1Path), np.load(oD2Path)
        innerValWdist, outer1ValWdist, outer2ValWdist = np.load(iVDPath), np.load(oVD1Path), np.load(oVD2Path)

        innerGLosses, outer1GLosses, outer2GLosses = np.load(iGPath), np.load(oG1Path), np.load(oG2Path)
        innerValGLosses, outer1ValGLosses, outer2ValGLosses = np.load(iVGPath), np.load(oVG1Path), np.load(oVG2Path)

        innerGradG, outer1GradG, outer2GradG = np.load(iGradGPath), np.load(oGradG1Path), np.load(oGradG2Path)
        innerGradD, outer1GradD, outer2GradD = np.load(iGradDPath), np.load(oGradD1Path), np.load(oGradD2Path)

        # KL plots
        def _plot_series(y, title, outname, ylog=True):
            plt.figure(dpi=200)
            plt.title(title)
            plt.grid(True, which='both', color='0.65', linestyle='-')
            plt.plot(y)
            if ylog:
                plt.yscale("log")
            plt.savefig(os.path.join(fig_path, outname))
            plt.close()

        _plot_series(innerKLDiv,  "Inner KL Divergence",  "inner_KL.png")
        _plot_series(outer1KLDiv, "Outer1 KL Divergence", "outer1_KL.png")
        _plot_series(outer2KLDiv, "Outer2 KL Divergence", "outer2_KL.png")

        # Critic W-distance (train vs val) – positive estimate; lower is better for generator
        def _plot_wdist(train, val, title, outname):
            plt.figure(dpi=200)
            plt.title(title)
            plt.grid(True, which='both', color='0.65', linestyle='-')
            plt.plot(train, label="training")
            plt.plot(val, label="validation")
            plt.yscale("log")
            plt.legend()
            plt.savefig(os.path.join(fig_path, outname))
            plt.close()

        _plot_wdist(innerWdist,  innerValWdist,  "Inner Critic W-distance",  "inner_Wdist.png")
        _plot_wdist(outer1Wdist, outer1ValWdist, "Outer1 Critic W-distance", "outer1_Wdist.png")
        _plot_wdist(outer2Wdist, outer2ValWdist, "Outer2 Critic W-distance", "outer2_Wdist.png")

        # Generator losses
        def _plot_gloss(tr, va, title, outname):
            plt.figure(dpi=200)
            plt.title(title)
            plt.grid(True, which='both', color='0.65', linestyle='-')
            plt.plot(tr, label="training")
            plt.plot(va, label="validation")
            plt.legend()
            plt.savefig(os.path.join(fig_path, outname))
            plt.close()

        _plot_gloss(innerGLosses,  innerValGLosses,  "Inner Generator Loss",  "inner_Gloss.png")
        _plot_gloss(outer1GLosses, outer1ValGLosses, "Outer1 Generator Loss", "outer1_Gloss.png")
        _plot_gloss(outer2GLosses, outer2ValGLosses, "Outer2 Generator Loss", "outer2_Gloss.png")

        # Gradient norms
        def _plot_grad(arr, title, outname):
            plt.figure(dpi=200)
            plt.title(title)
            plt.grid(True, which='both', color='0.65', linestyle='-')
            plt.plot(arr)
            plt.yscale("log")
            plt.savefig(os.path.join(fig_path, outname))
            plt.close()

        _plot_grad(innerGradG,  "Inner Generator ∥∇G∥₂",  "inner_GradG.png")
        _plot_grad(outer1GradG, "Outer1 Generator ∥∇G∥₂", "outer1_GradG.png")
        _plot_grad(outer2GradG, "Outer2 Generator ∥∇G∥₂", "outer2_GradG.png")

        _plot_grad(innerGradD,  "Inner Critic ∥∇D∥₂",  "inner_GradD.png")
        _plot_grad(outer1GradD, "Outer1 Critic ∥∇D∥₂", "outer1_GradD.png")
        _plot_grad(outer2GradD, "Outer2 Critic ∥∇D∥₂", "outer2_GradD.png")

    # Save DF?
    if save_df:
        noLeaksDF = pd.concat([innerDF, outer1DF, outer2DF], ignore_index=True)
        out_csv = os.path.join(run_dir, "noLeaksDF.csv")
        noLeaksDF.to_csv(out_csv, index=False)

    # Correlation plots & 1D hists
    if plot_results:
        Hxy, Het, Hrth, Hpp = make_plots(innerDF,  "inner",  run_id, 'inner',  run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(outer1DF, "outer1", run_id, 'outer1', run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(outer2DF, "outer2", run_id, 'outer2', run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(pd.concat([innerDF, outer1DF, outer2DF]),
                                         "outer1", run_id, 'noLeaks', run_dir)

        features = [' xx', ' yy', ' pxx', ' pyy', ' pzz', ' eneg', ' time', 'theta', ' rx', ' rp']
        chi2_tests = {'inner': {}, 'outer1': {}, 'outer2': {}, 'noLeaks': {}}
        chi2_inner = {' xx': 0, ' yy': 0, ' pxx': 0, ' pyy': 0,
                      ' pzz': 0, ' eneg': 0, ' time': 0, 'theta': 0, ' rx': 0}
        chi2_outer1 = chi2_inner.copy()
        chi2_outer2 = chi2_inner.copy()
        chi2_noLeaks = chi2_inner.copy()

        for key in chi2_tests.keys():
            out_dir = os.path.join(fig_path, '1dHists', key)
            os.makedirs(out_dir, exist_ok=True)
            for feat in features:
                if key != 'noLeaks':
                    # data DF first, then generated DF in original function signature – keep as-is
                    exec("plot_1d(" + key + "Data," + key + "DF,feat,chi2_" + key + ", fig_path, key)")
                else:
                    plot_1d(pd.concat([innerData, outer1Data, outer2Data], ignore_index=True),
                            pd.concat([innerDF, outer1DF, outer2DF], ignore_index=True),
                            feat, chi2_noLeaks, fig_path, key)

    gc.collect()
