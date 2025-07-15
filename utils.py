import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import shutil
import subprocess
import dataset
from generator import Generator, Generator2
import time
import json
import os
import tqdm
from scipy.stats import kstest as ks
from scipy.stats import binned_statistic_dd
from numba import njit
import random
import psutil  # Change 7: import psutil for memory logging
import gc      # Change 7: import gc for garbage collection

# === Change 5: Global style setup moved outside function scope ===
plt.style.use('seaborn-v0_8-deep')  # moved from inside check_run
plt.rcParams.update({'font.size': 25})  # moved from inside check_run


def compute_sparse_histogram(data, bin_edges):
    hist, _, _2 = binned_statistic_dd(data.detach().cpu().numpy(), values=None, statistic='count', bins=bin_edges)
    print(f"Size of hist in bytes: {hist.nbytes}")
    return torch.tensor(hist, dtype=torch.float32)


def get_kld(real: torch.Tensor,
            fake: torch.Tensor,
            bins: int = 12,
            epsilon: float = 1e-12) -> torch.Tensor:
    """
    Compute the KL-divergence between the distributions of real and fake samples.

    :param real: Tensor of shape (n_samples, n_features)
    :param fake: Tensor of shape (n_samples, n_features)
    :param bins: Number of bins per feature dimension (int) or sequence of ints
    :param epsilon: Small value to avoid log(0)
    :return: Scalar tensor containing the KL-divergence
    """
    # Move data to CPU numpy
    real_np = real.detach().cpu().numpy()
    fake_np = fake.detach().cpu().numpy()

    n_features = real_np.shape[1]
    # Build bin edges per dimension from real data
    if isinstance(bins, int):
        bin_counts = [bins] * n_features
    else:
        assert len(bins) == n_features, "bins must be int or sequence matching features"
        bin_counts = bins

    bin_edges = [np.linspace(real_np[:, i].min(), real_np[:, i].max(), b + 1)
                 for i, b in enumerate(bin_counts)]

    # Compute multidimensional histograms
    real_hist, _ = np.histogramdd(real_np, bins=bin_edges, density=True)
    fake_hist, _ = np.histogramdd(fake_np, bins=bin_edges, density=True)

    # Flatten and normalize to form probability mass functions
    real_p = real_hist.flatten()
    fake_p = fake_hist.flatten()
    real_p = real_p / (real_p.sum() + epsilon)
    fake_p = fake_p / (fake_p.sum() + epsilon)

    # Add epsilon to avoid zeros
    real_p = real_p + epsilon
    fake_p = fake_p + epsilon

    # Convert to torch tensors
    real_t = torch.from_numpy(real_p).float()
    fake_t = torch.from_numpy(fake_p).float()

    # Compute KL divergence: sum P_real * log(P_real / P_fake)
    kl = torch.sum(real_t * (real_t.log() - fake_t.log()))
    return kl


def add_features(df, pdg):
    """
    Adds redundant features so that all possible features are present
    :param df: dataframe with base features : [x,y,r_p,phi_p,eneg,time]
    :param pdg: configured pdg number
    :return: doesn't return, just alters the df
    """

    # dict format is {pdg : particle object}

    mass = particle_dict[pdg].mass
    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)

    # exp = (df[' eneg'] + mass) ** 2 - mass ** 2 - df[' rp'] ** 2
    # df.drop(df[exp < 0].index, inplace=True)

    df[' rp'] = np.sqrt(df[' pxx'] ** 2 + df[' pyy'] ** 2)
    df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx']) + np.pi

    # df[' pzz'] = -np.sqrt((df[' eneg'] + mass) ** 2 - mass ** 2 - df[' rp'] ** 2)
    df[' eneg'] = (df[' rp']**2+df[' pzz']**2)/(np.sqrt(df[' rp']**2+df[' pzz']**2+mass**2)+mass)
    # df[' pxx'] = df[' rp'] * np.cos(df[' phi_p'] - np.pi)
    # df[' pyy'] = df[' rp'] * np.sin(df[' phi_p'] - np.pi)
    # exp2 = df[' pzz'] / np.sqrt((df[' eneg'] + mass) ** 2 - mass ** 2)
    # df['theta'] = np.arccos(df[' pzz'] / np.sqrt((df[' eneg'] + mass) ** 2 - mass ** 2))
    df['theta'] = np.arctan2(df[' rp'], df[' pzz'])


def plot_correlations(x, y, xlabel, ylabel, run_id, key,
                      bins=[400, 400], loglog=False, Xlim=None, Ylim=None, path=None):
    H, xb, yb = np.histogram2d(x, y, bins=bins, range=[[x.min(), x.max()], [y.min(), y.max()]], density=True)
    X, Y = np.meshgrid(xb, yb)
    plt.figure(dpi=200)
    if xlabel == "x[mm]":
        vmin, vmax = 1e-10, 1e-6
    elif xlabel == "t[ns]":
        vmin, vmax = 1e-6, 1e3
    elif xlabel == '\phi_p [rad]':
        vmin, vmax = 1, 1e-4
    else:
        vmin, vmax = 1e-6, 1e-3
    plt.pcolormesh(X, Y, H.T, norm="log", vmin=vmin, vmax=vmax)
    # plt.tick_params()
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
        hist_path = path + 'plots/2dHists'
        if not os.path.isdir(hist_path + '/' + key):
            if not os.path.isdir(hist_path):
                os.mkdir(hist_path)
            os.mkdir(hist_path + '/' + key)
        clean_xlabel = ''.join(char for char in xlabel if char.isalnum())
        clean_ylabel = ''.join(char for char in ylabel if char.isalnum())
        plt.savefig(hist_path + '/' + key + '/2d-' + clean_xlabel + '-' + clean_ylabel + '.png', bbox_inches='tight')
        plt.close()
    else:
        hist_path = "/storage/agrp/alonle/GAN_Output"
        plt.savefig(hist_path + '/' + key + xlabel + '-' + ylabel + '.png')
        plt.close()
    return H


def make_plots(df, dataGroup, run_id="", key="", path=""):
    """
    Takes a dataframe and its data group and makes correlation plots for x-y,E-t,r_x-theta,phi_x-phi_p
    :param df: dataframe containing both polar and cartesian forms of data
    :param dataGroup: inner/outer
    :param run_id: string
    :return: null
    """
    # BEAM TRIM
    # x_lim = [-4500, 1500]
    # y_lim = [-3000, 6000]
    # # R CUT
    x_lim = [-4000, 4000]
    y_lim = [-4000, 4000]
    if dataGroup == 'inner':
        x_lim = [-1700, 500]
        y_lim = [-2500, 500]

    Hxy = plot_correlations(df[' xx'], df[' yy'], 'x[mm]', 'y[mm]', run_id, key, path=path)  # , Xlim=x_lim, Ylim=y_lim
    energy_bins = 10 ** np.linspace(-13, 0, 400)
    bin_stop = np.log10(np.max(df[' time']))
    time_bins = 10 ** np.linspace(1, bin_stop+0.5, 400)
    Het = plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', run_id, key, bins=[time_bins, energy_bins],
                            loglog=True, path=path) #, Xlim=10**6.5
    Hrth = plot_correlations(df[' rx'], df['theta'], 'r [mm]', '\\theta_p [rad]', run_id, key, path=path)
    Hpp = plot_correlations(df[' phi_p'], df[' phi_x'], '\phi_p [rad]', '\phi_x [rad]', run_id, key, path=path)
    return Hxy, Het, Hrth, Hpp


def split(df):
    """
    splits a dataframe into regions I, II, III according to their xy position
    :param df: input dataframe, contains all particles
    :return: inner dataframe, outer dataframe
    """
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


def make_polar_features(df):
    df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)
    df[' rp'] = np.sqrt(df[' pxx'] ** 2 + df[' pyy'] ** 2)
    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx']) + np.pi
    df['theta'] = np.arccos(df[' pzz'] / np.sqrt(df[' pzz'] ** 2 + df[' rp'] ** 2))


def get_q(ds):
    """
    get quantile array from dataset
    :param ds: ParticleDataset object
    :return: quantiles used for dataset
    """
    return ds.quantiles.quantiles_


def plot_features(ds, save_path=None):
    """Plots histograms and feature diagnostics for a dataset in one figure."""
    features = ds.preprocess.columns
    n_features = len(features)
    fig, axes = plt.subplots(4, n_features, figsize=(5 * n_features, 12))
    # fig.suptitle("Feature Diagnostics", fontsize=16)
    n_events = ds.preprocess.shape[0]
    generated_qt_df = np.random.randn(n_events, n_features)
    empty_arr = np.empty((0, n_features))
    ds2 = dataset.ParticleDataset(ds.cfg)
    ds2.data = pd.DataFrame(empty_arr, columns=features)
    data_values = ds2.quantiles.inverse_transform(generated_qt_df)
    for i, feature in enumerate(features):
        ds2.data[feature] = data_values[:, i]
    ds2.apply_transformation(ds2.cfg, inverse=True)
    xlabels = features
    print("Feature       Slope       Curvature")
    print("-" * 40)

    for i, col in enumerate(ds.preprocess.columns):
        # Extract axes
        ax1, ax2, ax3, ax4 = axes[:, i]
        # , ax5
        # Data for plotting
        norm = np.max(get_q(ds)[:, i])
        slope = np.max(get_q(ds)[1:, i] - get_q(ds)[:-1, i]) / norm
        curvature = np.max(get_q(ds)[2:, i] - 2 * get_q(ds)[1:-1, i] + get_q(ds)[:-2, i]) / norm

        print(f"{col:<10}  {slope:.4f}    {curvature:.4f}")

        # Plot histograms and quantiles
        if col == ' eneg':
            bins = 10 ** np.linspace(-12, 0, 400)
        elif col == ' time':
            bins = 10 ** np.linspace(1, 6.5, 400)
        else:
            bins = 400

        ax1.hist(ds.preprocess[col], bins=bins, log=True, color="#4682b4")
        ax2.hist(ds.preqt[:, i], bins=400, log=True, color="#4682b4")
        ax3.hist(ds.data[:, i], bins=400, log=True, color="#4682b4")
        ax4.plot(ds.quantiles.quantiles_[:, i], label="Quantiles", color="#4682b4")

        # ax5.hist(ds.preprocess[col], bins=bins, log=True, alpha=0.6)
        # ax5.hist(ds2.data[col], bins=bins, log=True, alpha=0.6)
        # Add titles and labels
        # if i == 0:  # Add column headers for the first row
        #     ax1.set_title("Raw Data")
        #     ax2.set_title("Normalized Features")
        #     ax3.set_title("Quantile Transformation")
        #     ax4.set_title("Approximated \nQuantile Function")
        #     ax5.set_title("QT-generated Data")
        fontsize=28
        if i == 0:
            ax1.set_ylabel(f"frequency", fontsize=fontsize)
            ax2.set_ylabel(f"frequency", fontsize=fontsize)
            ax3.set_ylabel(f"frequency", fontsize=fontsize)
            ax4.set_ylabel(f"data value", fontsize=fontsize)
        ax1.set_xlabel(f"{xlabels[i]}", fontsize=fontsize)
        ax2.set_xlabel(f"{xlabels[i]}", fontsize=fontsize)
        ax4.set_xlabel(f"Quantile", fontsize=fontsize)
        # if col in [' eneg', ' time']:
        # _____ ax5.set_xscale('log')
        for ax in [ax1, ax2, ax3, ax4]:  # , ax5
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.grid(True, alpha=0.6)

        # ax5.legend(["Original Data", "Inverse QT"])
    plt.tight_layout()  # Leave space for the overall title
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    add_features(ds2.data, ds2.cfg["pdg"])

    return ds2.data, ds.preprocess


def generate_ds(generator_net, factor, cfg):
    """
    :param generator_net:
    :param factor:
    :param cfg:
    :return: numpy array of generated data
    """
    ds = dataset.ParticleDataset(cfg)
    numEvents = np.shape(ds.data)[0]
    generator_net.eval().to('cpu')
    generated_data = generator_net(torch.randn(np.int64(numEvents / factor), cfg['noiseDim'], device='cpu'))
    generated_data = generated_data.detach().numpy()
    features = cfg['features'].keys()
    ds.data = pd.DataFrame(np.empty((0, len(features))), columns=features)
    data_values = ds.quantiles.inverse_transform(generated_data) if ds.quantiles is not None else generated_data
    for i, feature in enumerate(features):
        ds.data[feature] = data_values[:, i]
    del data_values
    ds.apply_transformation(cfg, inverse=True)
    fake, real = ds.data, ds.preprocess
    del ds
    gc.collect()
    return fake, real


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)
    elif isinstance(m, nn.InstanceNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)


def generate_fake_real_dfs(run_id, cfg, run_dir, generator_net=None):
    """
    Given a run id load the trained model in run_id dir to its respective trainer and return the generated dataframe
    :param generator_net: network, if available
    :param run_id: run id
    :param cfg: relevant config file
    :param run_dir: local is none / cluster is path to run dir
    :return:
    """

    # Create a generator based on the model's number of parameters used
    numFeatures = len(cfg["features"].keys())
    # Load parameters
    model_path = run_dir + cfg["dataGroup"] + "_Gen_model.pt"
    dim = cfg["noiseDim"]
    if generator_net is None:
        generator_net = nn.DataParallel(Generator(dim, numFeatures=numFeatures))
        try:
            generator_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except RuntimeError as e:
            print("Error loading model state_dict:", e)
            print("Switching model")
            generator_net = nn.DataParallel(Generator2(dim, numFeatures=numFeatures))
            generator_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # TODO: remove factor, find a different way to ease local data generation
    # Read data used for training
    fake_df, real_df = generate_ds(generator_net, factor=1, cfg=cfg)
    real_df = real_df[real_df[' time'] <= 1e6]
    fake_df = fake_df[fake_df[' time'] <= 1e6]
    add_features(fake_df, cfg['pdg'])
    add_features(real_df, cfg['pdg'])

    return fake_df, real_df


def check_run(run_id, path=None, calculate_BED=True, save_df=False, plot_metrics=True, plot_results=True):
    plt.ioff()
    if path is None:
        print("Local path")
        run_dir = 'Output/run_' + run_id + '/'
    else:
        print("Cluster path")
        run_dir = path + '/run_' + run_id + '/'
    fig_path = run_dir + 'plots'
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    fig_path += '/'

    with open(run_dir + "cfg_inner.json", 'r') as inner_file:
        cfg_inner = json.loads(inner_file.read())
    with open(run_dir + "cfg_outer1.json", 'r') as outer1_file:
        cfg_outer1 = json.loads(outer1_file.read())
    with open(run_dir + "cfg_outer2.json", 'r') as outer2_file:
        cfg_outer2 = json.loads(outer2_file.read())

    # Specifically for data and norm files, changes the path to local
    if path is None:
        fix_path(cfg_inner, "data_path")
        fix_path(cfg_outer1, "data_path")
        fix_path(cfg_outer2, "data_path")

    if plot_results or save_df or calculate_BED:
        generation_time_a = time.localtime()
        innerDF, innerData = generate_fake_real_dfs(run_id, cfg_inner, run_dir)
        outer1DF, outer1Data = generate_fake_real_dfs(run_id, cfg_outer1, run_dir)
        outer2DF, outer2Data = generate_fake_real_dfs(run_id, cfg_outer2, run_dir)
        generation_time_b = time.localtime()
        print(f'Created DFs in {get_time(generation_time_a, generation_time_b)}')

    if calculate_BED:
        print("getting batch ED...")
        batch_size = 1000
        small_batch = 50
        inner_null_values, inner_H1_values = get_batch_ed_histograms(
            innerDF, innerData,
            batch_size=batch_size)
        outer1_null_values, outer1_H1_values = get_batch_ed_histograms(
            outer1DF, outer1Data,
            batch_size=batch_size)
        outer2_null_values, outer2_H1_values = get_batch_ed_histograms(
            outer2DF, outer2Data,
            batch_size=small_batch)
        make_ed_fig(inner_null_values, inner_H1_values, 'inner', fig_path)
        make_ed_fig(outer1_null_values, outer1_H1_values, 'outer1', fig_path)
        make_ed_fig(outer2_null_values, outer2_H1_values, 'outer2', fig_path)
        print(f"[mem after ED init] {psutil.Process().memory_info().rss / 1e9:.2f} GB")

    if plot_metrics:
        # ─── file paths (new naming) ─────────────────────────────────────────────
        iKLPath = os.path.join(run_dir, "KL_inner.npy")
        oKL1Path = os.path.join(run_dir, "KL_outer1.npy")
        oKL2Path = os.path.join(run_dir, "KL_outer2.npy")

        iDPath = os.path.join(run_dir, "D_inner.npy")
        oD1Path = os.path.join(run_dir, "D_outer1.npy")
        oD2Path = os.path.join(run_dir, "D_outer2.npy")

        iVDPath = os.path.join(run_dir, "ValD_inner.npy")
        oVD1Path = os.path.join(run_dir, "ValD_outer1.npy")
        oVD2Path = os.path.join(run_dir, "ValD_outer2.npy")

        iGPath = os.path.join(run_dir, "G_inner.npy")
        oG1Path = os.path.join(run_dir, "G_outer1.npy")
        oG2Path = os.path.join(run_dir, "G_outer2.npy")

        iVGPath = os.path.join(run_dir, "ValG_inner.npy")
        oVG1Path = os.path.join(run_dir, "ValG_outer1.npy")
        oVG2Path = os.path.join(run_dir, "ValG_outer2.npy")

        iGradGPath = os.path.join(run_dir, "GradG_inner.npy")
        oGradG1Path = os.path.join(run_dir, "GradG_outer1.npy")
        oGradG2Path = os.path.join(run_dir, "GradG_outer2.npy")

        iGradDPath = os.path.join(run_dir, "GradD_inner.npy")
        oGradD1Path = os.path.join(run_dir, "GradD_outer1.npy")
        oGradD2Path = os.path.join(run_dir, "GradD_outer2.npy")

        # ─── load arrays ────────────────────────────────────────────────────────────
        innerKLDiv = np.load(iKLPath)
        outer1KLDiv = np.load(oKL1Path)
        outer2KLDiv = np.load(oKL2Path)

        innerWdist = np.load(iDPath)
        outer1Wdist = np.load(oD1Path)
        outer2Wdist = np.load(oD2Path)
        innerValWdist = np.load(iVDPath)
        outer1ValWdist = np.load(oVD1Path)
        outer2ValWdist = np.load(oVD2Path)

        innerGLosses = np.load(iGPath)
        outer1GLosses = np.load(oG1Path)
        outer2GLosses = np.load(oG2Path)
        innerValGLosses = np.load(iVGPath)
        outer1ValGLosses = np.load(oVG1Path)
        outer2ValGLosses = np.load(oVG2Path)

        innerGradG = np.load(iGradGPath)
        outer1GradG = np.load(oGradG1Path)
        outer2GradG = np.load(oGradG2Path)

        innerGradD = np.load(iGradDPath)
        outer1GradD = np.load(oGradD1Path)
        outer2GradD = np.load(oGradD2Path)

        # ─── plot KL divergences ────────────────────────────────────────────────────
        plt.figure(dpi=200)
        plt.title("Inner KL Divergence")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(innerKLDiv)
        plt.yscale("log")
        plt.savefig(fig_path + "inner_KL.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer1 KL Divergence")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer1KLDiv)
        plt.yscale("log")
        plt.savefig(fig_path + "outer1_KL.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer2 KL Divergence")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer2KLDiv)
        plt.yscale("log")
        plt.savefig(fig_path + "outer2_KL.png")
        plt.close()

        # ─── plot Training vs Validation W-distance (Critic) ────────────────────────
        plt.figure(dpi=200)
        plt.title("Inner Critic W-distance")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(innerWdist, label="training")
        plt.plot(innerValWdist, label="validation")
        plt.legend()
        plt.savefig(fig_path + "inner_Wdist.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer1 Critic W-distance")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer1Wdist, label="training")
        plt.plot(outer1ValWdist, label="validation")
        plt.legend()
        plt.savefig(fig_path + "outer1_Wdist.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer2 Critic W-distance")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer2Wdist, label="training")
        plt.plot(outer2ValWdist, label="validation")
        plt.legend()
        plt.savefig(fig_path + "outer2_Wdist.png")
        plt.close()

        # ─── plot Generator Losses ──────────────────────────────────────────────────
        plt.figure(dpi=200)
        plt.title("Inner Generator Loss")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(innerGLosses, label="training")
        plt.plot(innerValGLosses, label="validation")
        plt.legend()
        plt.savefig(fig_path + "inner_Gloss.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer1 Generator Loss")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer1GLosses, label="training")
        plt.plot(outer1ValGLosses, label="validation")
        plt.legend()
        plt.savefig(fig_path + "outer1_Gloss.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer2 Generator Loss")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer2GLosses, label="training")
        plt.plot(outer2ValGLosses, label="validation")
        plt.legend()
        plt.savefig(fig_path + "outer2_Gloss.png")
        plt.close()

        # ─── plot Gradient Norms (per-batch, smoothed externally if desired) ────────
        plt.figure(dpi=200)
        plt.title("Inner Generator ∥∇G∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(innerGradG)
        plt.yscale("log")
        plt.savefig(fig_path + "inner_GradG.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer1 Generator ∥∇G∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer1GradG)
        plt.yscale("log")
        plt.savefig(fig_path + "outer1_GradG.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer2 Generator ∥∇G∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer2GradG)
        plt.yscale("log")
        plt.savefig(fig_path + "outer2_GradG.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Inner Critic ∥∇D∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(innerGradD)
        plt.yscale("log")
        plt.savefig(fig_path + "inner_GradD.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer1 Critic ∥∇D∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer1GradD)
        plt.yscale("log")
        plt.savefig(fig_path + "outer1_GradD.png")
        plt.close()

        plt.figure(dpi=200)
        plt.title("Outer2 Critic ∥∇D∥₂")
        plt.grid(True, which='both', color='0.65', linestyle='-')
        plt.plot(outer2GradD)
        plt.yscale("log")
        plt.savefig(fig_path + "outer2_GradD.png")
        plt.close()

    features = [' xx', ' yy', ' pxx', ' pyy', ' pzz', ' eneg', ' time', 'theta', ' rx', ' rp']
    chi2_tests = {'inner': {}, 'outer1': {}, 'outer2': {}, 'noLeaks': {}}
    chi2_inner = {' xx': 0, ' yy': 0, ' pxx': 0, ' pyy': 0,
                  ' pzz': 0, ' eneg': 0, ' time': 0, 'theta': 0, ' rx': 0}
    chi2_outer1 = chi2_inner.copy()
    chi2_outer2 = chi2_inner.copy()
    chi2_combined = chi2_inner.copy()
    chi2_noLeaks = chi2_inner.copy()
    posIn = (innerDF[' time'] <= 1e6) & (innerDF[' rx'] <= 4000) & (innerDF[' xx'] <= 500) & (innerDF[' xx'] >= -1700) & (innerDF[' yy'] <= 520)
    posOut1 = (outer1DF[' time'] <= 1e6) & (outer1DF[' rx'] <= 4000) & ((outer1DF[' xx'] >= 500) | (outer1DF[' yy'] >= 520))
    posOut2 = (outer2DF[' time'] <= 1e6) & (outer2DF[' rx'] <= 4000) & ((outer2DF[' xx'] < -1700) & (outer2DF[' yy'] <= 520))

    if save_df:
        noLeaksDF = pd.concat([innerDF[posIn], outer1DF[posOut1], outer2DF[posOut2]])
        noLeaksDF.to_csv(run_dir + "noLeaksDF.csv")

    if plot_results:
        Hxy, Het, Hrth, Hpp = make_plots(innerDF, "inner", run_id, 'inner', run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(outer1DF, "outer1", run_id, 'outer1', run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(outer2DF, "outer2", run_id, 'outer2', run_dir)
        Hxy, Het, Hrth, Hpp = make_plots(pd.concat([innerDF[posIn], outer1DF[posOut1], outer2DF[posOut2]]), "outer1", run_id, 'noLeaks', run_dir)
        # GHxy, GHet, GHrth, GHpp = make_plots(combinedDF, "outer1", run_id, 'combined', run_dir)

        for key in chi2_tests.keys():
            if not os.path.isdir(fig_path + '1dHists/' + key):
                if not os.path.isdir(fig_path + '1dHists'):
                    os.mkdir(fig_path + '1dHists')
                os.mkdir(fig_path + '1dHists/' + key)
            for feat in features:
                if not key == 'noLeaks':
                    exec("plot_1d(" + key + "Data," + key + "DF,feat,chi2_" + key + ", fig_path, key)")
                else:
                    plot_1d(pd.concat([innerData[posIn], outer1Data[posOut1], outer2Data[posOut2]]),
                            pd.concat([innerDF[posIn], outer1DF[posOut1], outer2DF[posOut2]])
                            , feat, chi2_noLeaks, fig_path, key)


def plot_1d(data, DF, feat, ks, fig_path, key):
    plt.figure(dpi=200)
    plt.yscale('log')
    bins = np.linspace(np.min(DF[feat]), np.max(DF[feat]), 200)
    if feat == ' time':
        bins = np.logspace(np.log10(np.min(np.abs(DF[feat]))), np.log10(np.sort(DF[feat])[-10]), 200)
        plt.xscale('log')
    elif feat == ' eneg':
        bins = np.logspace(np.log10(np.min(np.abs(DF[feat]))), np.log10(np.sort(DF[feat])[-10]), 200)
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
    plt.xlabel(labeldict[feat])
    plt.savefig(fig_path + '1dHists/' + key + '/' + feat.strip().capitalize(),bbox_inches='tight')
    plt.close()


def get_distance(data, DF, feat):
    bins = np.linspace(np.min(data[feat]), np.max(data[feat]), 1000)
    if feat == ' time':
        bins = np.linspace(np.min(data[feat]), np.sort(data[feat])[-10], 1000)
    elif feat == ' eneg':
        bins = np.linspace(np.min(data[feat]), np.sort(data[feat])[-10], 1000)
    h1 = np.histogram(data[feat], bins=bins, density=True)[0]
    h2 = np.histogram(DF[feat], bins=bins, density=True)[0]
    mean = 0
    for i in range(len(h1)):
        if h1[i] == 0 and h2[i] == 0:
            mean += 0
        else:
            mean += np.abs(h2[i] - h1[i]) / (h1[i] + h2[i])
    return mean / len(h1)


def get_time(end_time, beg_time=np.zeros(9)):
    """
    get time elapsed between two localtime() objects in format hh:mm:ss
    :param end_time:
    :param beg_time:
    :return:
    """
    return time.asctime(time.struct_time(np.abs(np.int64(end_time) - np.int64(beg_time))))[11:19]


# calculates the Euclidean distance matrix for two np arrays x and y
@njit
def euclidean_distance_matrix(x, y, out):
    n_x, dim = x.shape
    n_y = y.shape[0]
    # original:
    # D_XY = np.zeros((n_x, n_y))
    for i in range(n_x):
        for j in range(n_y):
            s = 0.0
            for k in range(dim):
                s += (x[i, k] - y[j, k]) ** 2
            out[i, j] = np.sqrt(s)
    return out


def get_batch_ed_histograms(x_c, y_c, batch_size=1000):
    """
    NORMALIZES THE DATA x = (x - np.mean(x)) / np.std(x)
    get histograms of energy distance for batches of data with itself (null) and data with generated data (h1)
    :param x: data in the shape [num_samples, num_features]
    :param y: data in the shape [num_samples, num_features]
    :param batch_size:
    :return: null values, H1 values
    """

    x = x_c.copy()
    y = y_c.copy()

    for f in x.columns:
        x[f] = (x[f] - np.mean(x[f])) / np.std(x[f])
        y[f] = (y[f] - np.mean(y[f])) / np.std(y[f])

        #     # Just so that any of the features won't take over the metric
        #     if f in [' rx', ' rp', ' eneg', ' time']:
        #         x[f] = np.log(x[f])
        #         y[f] = np.log(y[f])
        #

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


def get_sampled_ed_histograms(x, y, batch_size=1000, num_iters=10000):
    """
    NORMALIZES THE DATA x = (x - np.mean(x)) / np.std(x)
    get histograms of energy distance for batches of data with itself (null) and data with generated data (h1)
    :param x: data in the shape [num_samples, num_features]
    :param y: data in the shape [num_samples, num_features]
    :param batch_size:
    :return: null values, H1 values
    """
    # Process the data once before sampling loop
    x_processed = x.copy()
    y_processed = y.copy()

    for f in x.columns:
        # Just so that any of the features won't take over the metric
        # if f in [' rx', ' rp', ' eneg', ' time']:
        #     x_processed[f] = np.log(x_processed[f])
        #     y_processed[f] = np.log(y_processed[f])

        x_processed[f] = (x_processed[f] - np.mean(x_processed[f])) / np.std(x_processed[f])
        y_processed[f] = (y_processed[f] - np.mean(y_processed[f])) / np.std(y_processed[f])

    # Pre-generate all random indices for sampling
    x_indices = np.random.choice(len(x_processed), size=(num_iters, batch_size), replace=True)
    y_indices = np.random.choice(len(y_processed), size=(num_iters, batch_size), replace=True)
    y_prime_indices = np.random.choice(len(y_processed), size=(num_iters, batch_size), replace=True)

    # Convert to numpy arrays for faster indexing
    x_values = x_processed.values
    y_values = y_processed.values

    ED_null = np.zeros(num_iters)
    ED_H1 = np.zeros(num_iters)

    # Vectorize the energy distance calculation if possible
    for i in tqdm.tqdm_notebook(range(num_iters), desc='Calculating ED', leave=False):
        x_batch = x_values[x_indices[i]]
        y_batch = y_values[y_indices[i]]
        y_prime_batch = y_values[y_prime_indices[i]]

        ED_null[i] = get_ed(x_batch, y_batch)
        ED_H1[i] = get_ed(y_prime_batch, y_batch)

    return ED_null, ED_H1


# original get_batches:
def get_batches(array, batch_size):
    batch_list = []
    i = 0
    while i + batch_size < array.shape[0]:
        batch_list.append(array[i:i + batch_size, :])
        i += batch_size
    return batch_list


def get_ed(x, y):
    n_x = np.shape(x)[0]
    n_y = np.shape(y)[0]
    buf = np.empty((x.shape[0], x.shape[0]), dtype=np.float32)
    D_XX = euclidean_distance_matrix(x, x, out=buf)
    XX_mean = np.sum(D_XX) / n_x ** 2
    del D_XX
    buf = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
    D_XY = euclidean_distance_matrix(x, y, out=buf)
    XY_mean = np.sum(D_XY) / (n_x * n_y)
    del D_XY
    buf = np.empty((y.shape[0], y.shape[0]), dtype=np.float32)
    D_YY = euclidean_distance_matrix(y, y, out=buf)
    YY_mean = np.sum(D_YY) / n_y ** 2
    del D_YY
    del buf
    return 2 * XY_mean - XX_mean - YY_mean


def save_cfg(cfg):
    inner_obj = json.dumps(cfg, indent=12)

    # Writing to sample.json
    with open(cfg['outputDir'] + "cfg_" + cfg['dataGroup'] + ".json", "w") as outfile:
        outfile.write(inner_obj)


def fix_path(cfg, feature):
    """
    TEMPORARY, just to fix paths when going back from cluster to local
    :param cfg:
    :param feature:
    :return:
    """
    file_name = cfg[feature].split("/")[-1]
    cfg[feature] = "TrainData/" + file_name


def is_latex_available():
    latex = shutil.which("latex")
    dvipng = shutil.which("dvipng")
    gs = shutil.which("gs")  # Ghostscript as fallback
    return latex is not None and (dvipng is not None or gs is not None)


def make_ed_fig(null, H1, group, fig_path, real_tag="Y", fake_tag="X", plotting=True):
    ks_test = ks(null, H1)
    pval, ks_stat = ks_test.pvalue, ks_test.statistic
    if plotting:
        fig, axs = plt.subplots(dpi=200)
        if is_latex_available():
            try:
                # Check if LaTeX is functional
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
            plt.text(.98, .5, f' $p$-value: ${compact_latex(pval)}$ ', ha='right', va='top', transform=axs.transAxes)
            plt.text(.98, .4, f' $K_{{n,m}}={ks_stat:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
        else:
            plt.text(.98, .5, f' $p$-value: ${pval:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
            plt.text(.98, .4, f' $K_{{n,m}}={ks_stat:.2f}$ ', ha='right', va='top', transform=axs.transAxes)
        plt.xlabel("$D_E \\rm{[a.u]}$")
        plt.ylabel("frequency")
        plt.yscale('log')
        axs.legend([f"$D_E({real_tag},{fake_tag})$", f"$D_E({real_tag},{real_tag}^\prime)$"])
        fig.savefig(fig_path + group + '_histograms.png', bbox_inches='tight')
    return pval, ks_stat


def compact_latex(n):
    # Convert the number to scientific notation
    sci_notation = "{:.2e}".format(n).split('e')

    # Extract the coefficient and exponent
    coefficient = sci_notation[0]
    exponent = int(sci_notation[1])

    # Construct the LaTeX string
    latex_str = f"{coefficient} \\cdot 10^{{{exponent}}}"

    return latex_str


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
        # Return features as a list
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

# Convert pdg_dict to a dictionary of Particle instances
particle_dict = {
    pdg: Particle(name, mass, charge, spin, isospin, pdg)
    for pdg, (name, mass, charge, spin, isospin) in pdg_dict.items()
}

###
# ARCHIVE

# def plot_features(ds):
#     print("Feature Slope      Curvature")
#
#     for i, col in enumerate(ds.preprocess):
#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
#         fig.suptitle(col)
#
#         ax1.hist(ds.preprocess[col], bins=400)
#         ax2.hist(ds.preqt[:, i], bins=400)
#         ax3.hist(ds.data[:, i], bins=400)
#         ax4.plot(ds.quantiles.quantiles_[:, i])
#
#         feature = '{:8}'.format(col)
#
#         norm = np.max(get_q(ds)[:, i])
#         slope = np.max(get_q(ds)[1:, i] - get_q(ds)[:-1, i]) / norm
#         curvature = np.max(get_q(ds)[2:, i] - 2 * get_q(ds)[1:-1, i] + get_q(ds)[:-2, i]) / norm
#
#         print(feature + '%.4f     %.4f' % (slope, curvature))
#         print("\n")
#
#         ax1.set_yscale('log')
#         ax2.set_yscale('log')
#         ax3.set_yscale('log')
#         plt.show()

#
# def combine(innerT, outerT, real_df=None, inner=None, outer=None):
#     if real_df is not None:
#         inner, outer = split(real_df)
#     numEvents = (len(inner) + len(outer)) / 10
#     q_in = len(inner) / (len(inner) + len(outer))
#     inner_events = np.int64(np.floor(numEvents * q_in))
#     outer_events = np.int64(np.ceil(numEvents * (1 - q_in)))
#     print(inner_events, outer_events)
#     inner_df = generate_df(innerT, innerT.noiseDim, inner_events)
#     outer_df = generate_df(outerT, outerT.noiseDim, outer_events)
#
#     generated_df = pd.concat((inner_df, outer_df), axis=0)
#
#     return generated_df
#
# def get_perm_p_value(x, y, n_permutations, log_norm=True, progress=True):
#     if log_norm:
#         x_norm = np.copy(x)
#         x_norm = prep_matrix(x_norm)
#         y_norm = np.copy(y)
#         y_norm = prep_matrix(y_norm)
#     else:
#         x_norm = np.copy(x)
#         y_norm = np.copy(y)
#
#     beg_time = time.localtime()
#     D_XX = euclidean_distance_matrix(x_norm, x_norm)
#     D_XY = euclidean_distance_matrix(x_norm, y_norm)
#     D_YX = D_XY.T
#     D_YY = euclidean_distance_matrix(y_norm, y_norm)
#     n_x = np.shape(x)[1]
#     n_y = np.shape(y)[1]
#     ED_observed = 2 * np.sum(D_XY) / (n_x * n_y) - np.sum(D_XX) / n_x ** 2 - np.sum(D_YY) / n_y ** 2
#     end_time = time.localtime()
#     # print("Calculated ED in ", get_time(beg_time,end_time))
#     null_hyp_ED_dist = np.zeros(n_permutations)
#     if progress:
#         permutations_iter = tqdm.tqdm_notebook(range(n_permutations), desc='permutation')
#     else:
#         permutations_iter = range(n_permutations)
#     for i in permutations_iter:
#         [i_1, i_2, i_1s, i_2s,
#          j_1, j_2, j_1s, j_2s] = permutation_indices(n_x, n_y)
#
#         D_XXs = np.zeros_like(D_XX)
#         D_XXs[np.ix_(i_1s, i_1s)] = D_XX[np.ix_(i_1, i_1)]
#         D_XXs[np.ix_(i_1s, i_2s)] = D_XY[np.ix_(i_1, i_2)]
#         D_XXs[np.ix_(i_2s, i_1s)] = D_YX[np.ix_(i_2, i_1)]
#         D_XXs[np.ix_(i_2s, i_2s)] = D_YY[np.ix_(i_2, i_2)]
#         D_YYs = np.zeros_like(D_YY)
#         D_YYs[np.ix_(j_1s, j_1s)] = D_XX[np.ix_(j_1, j_1)]
#         D_YYs[np.ix_(j_1s, j_2s)] = D_XY[np.ix_(j_1, j_2)]
#         D_YYs[np.ix_(j_2s, j_1s)] = D_YX[np.ix_(j_2, j_1)]
#         D_YYs[np.ix_(j_2s, j_2s)] = D_YY[np.ix_(j_2, j_2)]
#         D_XYs = np.zeros_like(D_XY)
#         D_XYs[np.ix_(i_1s, j_1s)] = D_XX[np.ix_(i_1, j_1)]
#         D_XYs[np.ix_(i_1s, j_2s)] = D_XY[np.ix_(i_1, j_2)]
#         D_XYs[np.ix_(i_2s, j_1s)] = D_YX[np.ix_(i_2, j_1)]
#         D_XYs[np.ix_(i_2s, j_2s)] = D_YY[np.ix_(i_2, j_2)]
#         null_hyp_ED_dist[i] = 2 * np.mean(D_XYs) - np.mean(D_XXs) - np.mean(D_YYs)
#     p_value = (n_permutations - np.sum(null_hyp_ED_dist >= ED_observed)) / (n_permutations + 1)
#     return p_value, null_hyp_ED_dist, ED_observed
#
# # shuffles indexes based on permutation
# def permutation_indices(n_x, n_y):
#     n = n_x + n_y
#     idx_w = np.arange(n)
#     idx_g = np.concatenate((np.arange(n_x), np.arange(n_y)))
#     idx_p1 = np.random.choice(n_x + n_y, n_x, replace=False)
#     idx_p2 = np.setdiff1d(idx_w, idx_p1)
#     i_1 = idx_g[idx_p1[idx_p1 < n_x]]
#     i_1s = np.arange(len(i_1))
#     i_2 = idx_g[idx_p1[idx_p1 >= n_x]]
#     i_2s = np.arange(len(i_1), n_x)
#     j_1 = idx_g[idx_p2[idx_p2 < n_x]]
#     j_1s = np.arange(len(j_1))
#     j_2 = idx_g[idx_p2[idx_p2 >= n_x]]
#     j_2s = np.arange(len(j_1), n_y)
#     return i_1, i_2, i_1s, i_2s, j_1, j_2, j_1s, j_2s
#
# # takes log of energy and time, normalizes everything
# def prep_matrix(x):
#     x[[2, 4, 5], :] = np.log(x[[2, 4, 5], :])
#
#     for d in range(np.shape(x)[0]):
#         x[d, :] = x[d, :] - np.min(x[d, :]) / (np.max(x[d, :]) - np.min(x[d, :]))
# #     return x


# def generate_df(generator_net, numEvents, cfg):
#     """
#     :param generator_net:
#     :param numEvents:
#     :param cfg:
#     :return: numpy array of generated data
#     """
#
#     ds = dataset.ParticleDataset(cfg)
#
#     if cfg['applyQT']:
#         noise = torch.randn(numEvents, cfg['noiseDim'], device='cpu')
#     else:
#         print("here")
#         noise = np.float32(np.random.randn(numEvents, cfg['noiseDim']))
#         noise = ds.quantiles.inverse_transform(noise)
#         noise = torch.from_numpy(noise)
#         noise = noise.to('cpu')
#
#     generator_net.to('cpu')
#     generated_data = generator_net(noise)
#     generated_data = generated_data.detach().numpy()
#
#     features = cfg['features'].keys()
#     empty_arr = np.empty((0, len(features)))
#     ds.data = pd.DataFrame(empty_arr, columns=features)
#     data_values = ds.quantiles.inverse_transform(generated_data) if cfg['applyQT'] else generated_data
#     for i, feature in enumerate(features):
#         ds.data[feature] = data_values[:, i]
#     ds.apply_transformation(cfg, inverse=True)
#     generated_df = ds.data.copy()
#     add_features(generated_df, cfg["pdg"])
#     return generated_df

