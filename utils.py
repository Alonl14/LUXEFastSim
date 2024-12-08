import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import dataset
from generator import Generator
import time
import json
import os
import tqdm
from scipy.stats import kstest as ks
from scipy.stats import binned_statistic_dd


def compute_sparse_histogram(data, bin_edges):
    hist, _, _2 = binned_statistic_dd(data.detach().cpu().numpy(), values=None, statistic='count', bins=bin_edges)
    print(f"Size of hist in bytes: {hist.nbytes}")
    return torch.tensor(hist, dtype=torch.float32)


def get_kld(real, fake, bins=12):
    """
    Creates N-dimensional pdfs and calculates their KL-divergence
    :param real: real event features, shape: (n_events, features)
    :param fake: events generated by the net
    :param bins: number of bins for the histograms
    :return: KL-divergence associated with the pdfs
    """

    # Calculate bin edges based on real data
    min_vals = torch.min(real, dim=0)[0]
    max_vals = torch.max(real, dim=0)[0]
    print(real.size(), fake.size())
    # Create equally spaced bins for each dimension
    bin_edges = [torch.linspace(min_vals[i], max_vals[i], bins + 1) for i in range(real.shape[1])]

    # # Compute histograms using the same bin edges for real and fake data
    if len(min_vals) <= 7:
        target_hist = torch.histogramdd(real.cpu(), bins=bin_edges, density=True).hist
        current_hist = torch.histogramdd(fake.cpu(), bins=bin_edges, density=True).hist
    else:
        target_hist = compute_sparse_histogram(real, bin_edges)
        current_hist = compute_sparse_histogram(fake, bin_edges)

    # Normalize histograms to be probability densities
    target_hist = target_hist / torch.sum(target_hist)
    current_hist = current_hist / torch.sum(current_hist)

    # Add small epsilon to avoid zero values in log
    epsilon = 1e-9
    target_hist = target_hist + epsilon
    current_hist = current_hist + epsilon

    # Calculate KL divergence
    current_kld = torch.nn.functional.kl_div(
        torch.log(target_hist),  # Log of the target (real data)
        torch.log(current_hist),  # Current histogram (fake data)
        reduction='batchmean',  # Mean KL divergence over all bins
        log_target=True  # target_hist is already a log probability
    )

    return current_kld


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
    # TODO: The np.abs is TEMPORARY since network is under-trained,
    #       needs to be removed for future versions
    df[' pzz'] = -np.sqrt(np.abs((df[' eneg'] + mass) ** 2 - mass ** 2 - df[' rp'] ** 2))
    # df[' eneg'] = np.sqrt(df[' rp']**2+df[' pzz']**2+mass**2)-mass
    # df[' rp'] = np.sqrt((df[' eneg']+mass)**2-mass**2-df[' pzz']**2)
    # df[' rp'] = np.sqrt(df[' eneg']**2-df[' pzz']**2)
    # df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx'])+np.pi
    df[' pxx'] = df[' rp'] * np.cos(df[' phi_p'] - np.pi)
    df[' pyy'] = df[' rp'] * np.sin(df[' phi_p'] - np.pi)
    # TODO: The np.maximum is TEMPORARY since network is under-trained,
    #       needs to be removed for future versions
    df['theta'] = np.arccos(np.maximum(-1*np.ones_like(df[' pzz'].values),
                                       df[' pzz'] / np.sqrt((df[' eneg'] + mass) ** 2 - mass ** 2)))


def plot_correlations(x, y, xlabel, ylabel, run_id, key,
                      bins=[400, 400], loglog=False, Xlim=None, Ylim=None, path=None):
    H, xb, yb = np.histogram2d(x, y, bins=bins, range=[[x.min(), x.max()], [y.min(), y.max()]], density=True)
    X, Y = np.meshgrid(xb, yb)
    plt.figure(dpi=250)
    if xlabel == "x[mm]":
        vmin, vmax = 1e-10, 1e-6
    elif xlabel == "t[ns]":
        vmin, vmax = 1e-6, 1e3
    elif xlabel == 'phi_p [rad]':
        vmin, vmax = 1, 1e-4
    else:
        vmin, vmax = 1e-6, 1e-3
    plt.pcolormesh(X, Y, H.T, norm="log", vmin = vmin, vmax = vmax)

    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    if Xlim is not None:
        plt.xlim(Xlim)
    if Ylim is not None:
        plt.ylim(Ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.colorbar()
    if run_id is not None:
        hist_path = path + 'plots/2dHists'
        if not os.path.isdir(hist_path + '/' + key):
            if not os.path.isdir(hist_path):
                os.mkdir(hist_path)
            os.mkdir(hist_path + '/' + key)
        plt.savefig(hist_path + '/' + key + '/' + xlabel + '-' + ylabel + '.png')
    else:
        hist_path = "/storage/agrp/alonle/GAN_Output"
        plt.savefig(hist_path + '/' + key + xlabel + '-' + ylabel + '.png')
    return H


def make_plots(df, dataGroup, run_id=None, key=None, path=None):
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
    energy_bins = 10 ** np.linspace(-12, 0, 400)
    time_bins = 10 ** np.linspace(1, 8, 400)
    Het = plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', run_id, key, bins=[time_bins, energy_bins],
                            loglog=True, path=path)
    Hrth = plot_correlations(df[' rx'], df['theta'], 'r [mm]', 'theta_p [rad]', run_id, key, path=path)
    Hpp = plot_correlations(df[' phi_p'], df[' phi_x'], 'phi_p [rad]', 'phi_x [rad]', run_id, key, path=path)
    return Hxy, Het, Hrth, Hpp


def split(df):
    """
    splits a dataframe into inner and outer particles according to their xy position
    :param df: input dataframe, contains all particles
    :return: inner dataframe, outer dataframe
    """
    # -1700mm < x < 500mm and -2500mm < y < 500mm qualifies as inner
    # -1700mm > x,x < 500mm and y > 500mm qualifies as outer
    # The union of these 2 regions doesn't cover the initial area so prints percentage of discarded events

    pos = ((df[' xx'] < 500) * (df[' xx'] > -1700) * (df[' yy'] < 500))
    pos2 = ((df[' xx'] < 500) * (df[' xx'] > -1700) * (df[' yy'] < -2500))
    print(f'Discarded {np.sum(np.int64(pos2)) / len(pos)  : .4%} of the particles')
    inner_df = df[pos & (~pos2)]
    outer_df = df[(~pos) & (~pos2)]
    print(f'Inner particles : {len(inner_df) / len(df):.2%} \nOuter Particles : {len(outer_df) / len(df) : .2%}')
    return inner_df, outer_df


def make_norm_file(df, path):
    """
    makes a normalization file from the given file at path
    :param df: supply df to avoid reading potentially large dataframes
    :param path: path to df
    :return: path to normalization file
    """
    normDF = pd.DataFrame(columns=['max', 'min'], index=pd.Index(df.columns))
    for col in df.columns:
        normDF['max'][col] = np.max(df[col])
        normDF['min'][col] = np.min(df[col])
    normPath = path.replace('.csv', '_norm.csv')

    print(normDF)
    normDF.to_csv(normPath)
    print('New file created at: ' + normPath)
    return normPath


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


def plot_features(ds):
    print("Feature Slope      Curvature")

    for i, col in enumerate(ds.preprocess):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(col)

        ax1.hist(ds.preprocess[col], bins=400)
        ax2.hist(ds.preqt[:, i], bins=400)
        ax3.hist(ds.data[:, i], bins=400)
        ax4.plot(ds.quantiles.quantiles_[:, i])

        feature = '{:8}'.format(col)

        norm = np.max(get_q(ds)[:, i])
        slope = np.max(get_q(ds)[1:, i] - get_q(ds)[:-1, i]) / norm
        curvature = np.max(get_q(ds)[2:, i] - 2 * get_q(ds)[1:-1, i] + get_q(ds)[:-2, i]) / norm

        print(feature + '%.4f     %.4f' % (slope, curvature))
        print("\n")

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        plt.show()


def generate_df(generator_net, numEvents, cfg):
    """
    :param generator_net:
    :param numEvents:
    :param cfg:
    :return: numpy array of generated data
    """

    ds = dataset.ParticleDataset(cfg)

    if cfg['applyQT']:
        noise = torch.randn(numEvents, cfg['noiseDim'], device='cpu')
    else:
        print("here")
        noise = np.float32(np.random.randn(numEvents, cfg['noiseDim']))
        noise = ds.quantiles.inverse_transform(noise)
        noise = torch.from_numpy(noise)
        noise = noise.to('cpu')

    generator_net.to('cpu')
    generated_data = generator_net(noise)
    generated_data = generated_data.detach().numpy()

    features = cfg['features'].keys()
    empty_arr = np.empty((0, len(features)))
    ds.data = pd.DataFrame(empty_arr, columns=features)
    data_values = ds.quantiles.inverse_transform(generated_data) if cfg['applyQT'] else generated_data
    for i, feature in enumerate(features):
        ds.data[feature] = data_values[:, i]
    ds.apply_transformation(cfg, inverse=True)
    generated_df = ds.data.copy()
    add_features(generated_df, cfg["pdg"])
    return generated_df


def generate_ds(generator_net, factor, cfg):
    """
    :param generator_net:
    :param factor:
    :param cfg:
    :return: numpy array of generated data
    """
    ds = dataset.ParticleDataset(cfg)
    numEvents = np.shape(ds.data)[0]
    noise = torch.randn(np.int64(numEvents/factor), cfg['noiseDim'], device='cpu')
    generator_net.to('cpu')
    generated_data = generator_net(noise)
    generated_data = generated_data.detach().numpy()

    features = cfg['features'].keys()
    empty_arr = np.empty((0, len(features)))
    ds.data = pd.DataFrame(empty_arr, columns=features)
    data_values = ds.quantiles.inverse_transform(generated_data) if ds.quantiles is not None else generated_data
    for i, feature in enumerate(features):
        ds.data[feature] = data_values[:, i]
    ds.apply_transformation(cfg, inverse=True)
    return ds.data, ds.preprocess


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)
    elif isinstance(m, nn.InstanceNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)


def combine(innerT, outerT, real_df=None, inner=None, outer=None):
    if real_df is not None:
        inner, outer = split(real_df)
    numEvents = (len(inner) + len(outer)) / 10
    q_in = len(inner) / (len(inner) + len(outer))
    inner_events = np.int64(np.floor(numEvents * q_in))
    outer_events = np.int64(np.ceil(numEvents * (1 - q_in)))
    print(inner_events, outer_events)
    inner_df = generate_df(innerT, innerT.noiseDim, inner_events)
    outer_df = generate_df(outerT, outerT.noiseDim, outer_events)

    generated_df = pd.concat((inner_df, outer_df), axis=0)

    return generated_df


def generate_fake_real_dfs(run_id, cfg, run_dir, generator_net = None):
    """
    Given a run id load the trained model in run_id dir to its respective trainer and return the generated dataframe
    :param run_id: run id
    :param cfg: relevant config file
    :param run_dir: local is none / cluster is path to run dir
    :return:
    """

    # Create a generator based on the model's number of parameters used
    numFeatures = len(cfg["features"].keys())
    # cfg["noiseDim"] = numFeatures

    # Load parameters
    model_path = run_dir + cfg["dataGroup"] + "_Gen_model.pt"
    if generator_net is None:
        generator_net = nn.DataParallel(Generator(cfg["noiseDim"], numFeatures=numFeatures))
        try:
            generator_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except RuntimeError as e:
            print("Error loading model state_dict:", e)
            print("Switching model")
            generator_net = nn.DataParallel(Generator2(cfg["noiseDim"], numFeatures=numFeatures))
            generator_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # TODO: remove factor, find a different way to ease local data generation
    # Read data used for training
    fake_df, real_df = generate_ds(generator_net, factor=1, cfg=cfg)
    add_features(fake_df, cfg['pdg'])
    add_features(real_df, cfg['pdg'])

    return fake_df, real_df


def check_run(run_id, path=None):
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
        fix_path(cfg_inner, "norm_path")
        fix_path(cfg_outer1, "data_path")
        fix_path(cfg_outer1, "norm_path")
        fix_path(cfg_outer2, "data_path")
        fix_path(cfg_outer2, "norm_path")

    # TODO: Think of a different condition to check if a df is needed to be produced
    generation_time_a = time.localtime()
    innerDF, innerData = generate_fake_real_dfs(run_id, cfg_inner, run_dir)
    outer1DF, outer1Data = generate_fake_real_dfs(run_id, cfg_outer1, run_dir)
    outer2DF, outer2Data = generate_fake_real_dfs(run_id, cfg_outer2, run_dir)
    generation_time_b = time.localtime()
    print(f'Created DFs in {get_time(generation_time_a, generation_time_b)}')
    print("getting batch ED...")
    # TODO: for some reason null_values are nans
    min_length = min(len(innerDF), len(outer1DF), len(outer2DF))

    inner_null_values, inner_H1_values = get_batch_ed_histograms(
        innerDF.loc[:min(1e6, len(innerDF)-1), cfg_inner['features'].keys()],
        innerData.loc[:min(1e6, len(innerDF)-1), cfg_inner['features'].keys()],
        batch_size=100)
    outer1_null_values, outer1_H1_values = get_batch_ed_histograms(
        outer1DF.loc[:min(1e6, len(outer1DF)-1), cfg_outer1['features'].keys()],
        outer1Data.loc[:min(1e6, len(outer1DF)-1), cfg_outer1['features'].keys()],
        batch_size=100)
    outer2_null_values, outer2_H1_values = get_batch_ed_histograms(
        outer2DF.loc[:min(1e6, len(outer2DF)-1), cfg_outer2['features'].keys()],
        outer2Data.loc[:min(1e6, len(outer2DF)-1), cfg_outer2['features'].keys()],
        batch_size=100)
    make_ed_fig(inner_null_values, inner_H1_values, 'inner', False, fig_path)
    make_ed_fig(outer1_null_values, outer1_H1_values, 'outer1', True, fig_path)
    make_ed_fig(outer2_null_values, outer2_H1_values, 'outer2', True, fig_path)

    #######
    # else:
    #     innerDF = pd.read_csv(run_dir + 'innerDF.csv')
    #     outerDF = pd.read_csv(run_dir + 'outerDF.csv')
    ######

    iKLPath = run_dir + "KL_in.npy"
    oKL1Path = run_dir + "KL_out1.npy"
    oKL2Path = run_dir + "KL_out2.npy"

    iDLPath = run_dir + "D_losses_in.npy"
    oDL1Path = run_dir + "D_losses_out1.npy"
    oDL2Path = run_dir + "D_losses_out2.npy"

    iVDLPath = run_dir + "Val_D_losses_in.npy"
    oVDL1Path = run_dir + "Val_D_losses_out1.npy"
    oVDL2Path = run_dir + "Val_D_losses_out2.npy"

    iGLPath = run_dir + "G_losses_in.npy"
    oGL1Path = run_dir + "G_losses_out1.npy"
    oGL2Path = run_dir + "G_losses_out2.npy"
    iVGLPath = run_dir + "Val_G_losses_in.npy"
    oVGL1Path = run_dir + "Val_G_losses_out1.npy"
    oVGL2Path = run_dir + "Val_G_losses_out2.npy"

    innerKLDiv = np.load(iKLPath)
    outer1KLDiv = np.load(oKL1Path)
    outer2KLDiv = np.load(oKL2Path)

    innerDLosses = np.load(iDLPath)
    outer1DLosses = np.load(oDL1Path)
    outer2DLosses = np.load(oDL2Path)
    innerValDLosses = np.load(iVDLPath)
    outer1ValDLosses = np.load(oVDL1Path)
    outer2ValDLosses = np.load(oVDL2Path)

    innerGLosses = np.load(iGLPath)
    outer1GLosses = np.load(oGL1Path)
    outer2GLosses = np.load(oGL2Path)
    innerValGLosses = np.load(iVGLPath)
    outer1ValGLosses = np.load(oVGL1Path)
    outer2ValGLosses = np.load(oVGL2Path)

    plt.figure(dpi=200)
    plt.title("Inner KL divergence")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(innerKLDiv)
    plt.yscale('log')
    plt.savefig(fig_path + 'innerKLDiv.png')
    plt.figure(dpi=200)
    plt.title("Outer1 KL divergence")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer1KLDiv)
    plt.yscale('log')
    plt.savefig(fig_path + 'outer1KLDiv.png')
    plt.figure(dpi=200)
    plt.title("Outer2 KL divergence")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer2KLDiv)
    plt.yscale('log')
    plt.savefig(fig_path + 'outer2KLDiv.png')
    plt.figure(dpi=200)
    plt.title("Inner Critic Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(innerDLosses)
    plt.plot(innerValDLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'innerDLosses.png')
    plt.figure(dpi=200)
    plt.title("Outer1 Critic Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer1DLosses)
    plt.plot(outer1ValDLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'outer1DLosses.png')
    plt.figure(dpi=200)
    plt.title("Outer2 Critic Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer2DLosses)
    plt.plot(outer2ValDLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'outer2DLosses.png')

    plt.figure(dpi=200)
    plt.title("Inner Generator Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(innerGLosses)
    plt.plot(innerValGLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'innerGLosses.png')
    plt.figure(dpi=200)
    plt.title("Outer1 Generator Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer1GLosses)
    plt.plot(outer1ValGLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'outer1GLosses.png')
    plt.figure(dpi=200)
    plt.title("Outer2 Generator Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outer2GLosses)
    plt.plot(outer2ValGLosses)
    plt.legend(["training", "validation"])
    plt.savefig(fig_path + 'outer2GLosses.png')

    features = [' xx', ' yy', ' pxx', ' pyy', ' pzz', ' eneg', ' time', 'theta']
    chi2_tests = {'inner': {}, 'outer1': {}, 'outer2': {}, 'combined': {}, 'noLeaks': {}}
    dfDict = {'inner': {}, 'outer1': {}, 'outer2': {}, 'combined': {}, 'noLeaks': {}}
    chi2_inner = {' xx': 0, ' yy': 0, ' pxx': 0, ' pyy': 0,
                  ' pzz': 0, ' eneg': 0, ' time': 0, 'theta': 0}
    chi2_outer1 = chi2_inner.copy()
    chi2_outer2 = chi2_inner.copy()
    chi2_combined = chi2_inner.copy()
    chi2_noLeaks = chi2_inner.copy()
    combinedData = pd.concat([innerData, outer1Data, outer2Data])
    combinedDF = pd.concat([innerDF, outer1DF, outer2DF])
    noLeaksData = combinedData.copy()
    posIn = (innerDF[' rx'] <= 4000) & (innerDF[' xx'] < 500) & (innerDF[' xx'] > -1700) & (innerDF[' yy'] < 500)
    posOut1 = (outer1DF[' rx'] <= 4000) & ((outer1DF[' xx'] >= 500) | (outer1DF[' yy'] >= 500))
    posOut2 = (outer2DF[' rx'] <= 4000) & ((outer2DF[' xx'] < -1700) & (outer2DF[' yy'] <= 500))
    # outer1 = df_pdg[(df_pdg[' xx'] >= 500) | (df_pdg[' yy'] >= 500)]
    # outer2 = df_pdg[(df_pdg[' xx'] < -1700) & (df_pdg[' yy'] < 500)]
    # inner = df_pdg[(df_pdg[' xx'] < 500) & (df_pdg[' xx'] >= -1700) & (df_pdg[' yy'] < 500)]

    noLeakInner = innerDF[posIn]
    noLeakOuter1 = outer1DF[posOut1]
    noLeakOuter2 = outer2DF[posOut2]
    noLeaksDF = pd.concat([noLeakInner, noLeakOuter1, noLeakOuter2])

    # features_for_test = [' xx', ' yy', ' rp', ' phi_p', ' eneg', ' time']

    # noLeaks_null_values, noLeaks_H1_values = get_batch_ed_histograms(
    #     noLeaksDF.loc[:, features_for_test],
    #     combinedData.loc[:, features_for_test],
    #     batch_size=500)
    # make_ed_fig(noLeaks_null_values, noLeaks_H1_values, 'noLeaks', False, fig_path)

    dfDict['inner'] = innerDF
    dfDict['outer1'] = outer1DF
    dfDict['outer2'] = outer2DF
    dfDict['combined'] = combinedDF
    dfDict['noLeaks'] = noLeaksDF

    Hxy, Het, Hrth, Hpp = make_plots(innerDF, "inner", run_id, 'inner', run_dir)
    Hxy, Het, Hrth, Hpp = make_plots(outer1DF, "outer1", run_id, 'outer1', run_dir)
    Hxy, Het, Hrth, Hpp = make_plots(outer2DF, "outer2", run_id, 'outer2', run_dir)
    Hxy, Het, Hrth, Hpp = make_plots(noLeaksDF, "outer1", run_id, 'noLeaks', run_dir)
    GHxy, GHet, GHrth, GHpp = make_plots(combinedDF, "outer1", run_id, 'combined', run_dir)

    for key in chi2_tests.keys():
        if not os.path.isdir(fig_path + '1dHists/' + key):
            if not os.path.isdir(fig_path + '1dHists'):
                os.mkdir(fig_path + '1dHists')
            os.mkdir(fig_path + '1dHists/' + key)
        for feat in features:
            exec("plot_1d(" + key + "Data," + key + "DF,feat,chi2_" + key + ", fig_path, key)")
        exec("chi2_tests['" + key + "']=chi2_" + key)


def plot_1d(data, DF, feat, ks, fig_path, key):
    plt.figure(dpi=200)
    plt.yscale('log')
    bins = np.linspace(np.min(data[feat]), np.max(data[feat]), 400)
    if feat == ' time':
        bins = np.logspace(np.log10(np.min(data[feat])), np.log10(np.sort(data[feat]))[-10], 400)
        plt.xscale('log')
    elif feat == ' eneg':
        bins = np.logspace(np.log10(np.min(data[feat])), np.log10(np.sort(data[feat]))[-10], 400)
        plt.xscale('log')
    plt.text(.01, .85, 'distance = ' + f'{ks[feat]:.3f}', ha='left', va='top', transform=plt.gca().transAxes)
    plt.hist(DF[feat], bins=bins, density=True, alpha=0.6)
    plt.hist(data[feat], bins=bins, density=True, alpha=0.6)
    plt.legend(["Generated data", "FullSim data"])
    plt.title(feat)
    plt.savefig(fig_path + '1dHists/' + key + '/' + feat.strip().capitalize())


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


# takes log of energy and time, normalizes everything
def prep_matrix(x):
    x[[2, 4, 5], :] = np.log(x[[2, 4, 5], :])

    for d in range(np.shape(x)[0]):
        x[d, :] = x[d, :] - np.min(x[d, :]) / (np.max(x[d, :]) - np.min(x[d, :]))
    return x


# calculates the Euclidean distance matrix for two np arrays x and y
def euclidean_distance_matrix(x, y):
    D_XY = np.zeros([np.shape(x)[0], np.shape(y)[0]])
    shape = np.shape(D_XY)
    for i in range(shape[0]):
        for j in range(shape[1]):
            D_XY[i, j] = np.linalg.norm(x[i, :] - y[j, :])
    return D_XY


# shuffles indexes based on permutation
def permutation_indices(n_x, n_y):
    n = n_x + n_y
    idx_w = np.arange(n)
    idx_g = np.concatenate((np.arange(n_x), np.arange(n_y)))
    idx_p1 = np.random.choice(n_x + n_y, n_x, replace=False)
    idx_p2 = np.setdiff1d(idx_w, idx_p1)
    i_1 = idx_g[idx_p1[idx_p1 < n_x]]
    i_1s = np.arange(len(i_1))
    i_2 = idx_g[idx_p1[idx_p1 >= n_x]]
    i_2s = np.arange(len(i_1), n_x)
    j_1 = idx_g[idx_p2[idx_p2 < n_x]]
    j_1s = np.arange(len(j_1))
    j_2 = idx_g[idx_p2[idx_p2 >= n_x]]
    j_2s = np.arange(len(j_1), n_y)
    return i_1, i_2, i_1s, i_2s, j_1, j_2, j_1s, j_2s


def get_perm_p_value(x, y, n_permutations, log_norm=True, progress=True):
    if log_norm:
        x_norm = np.copy(x)
        x_norm = prep_matrix(x_norm)
        y_norm = np.copy(y)
        y_norm = prep_matrix(y_norm)
    else:
        x_norm = np.copy(x)
        y_norm = np.copy(y)

    beg_time = time.localtime()
    D_XX = euclidean_distance_matrix(x_norm, x_norm)
    D_XY = euclidean_distance_matrix(x_norm, y_norm)
    D_YX = D_XY.T
    D_YY = euclidean_distance_matrix(y_norm, y_norm)
    n_x = np.shape(x)[1]
    n_y = np.shape(y)[1]
    ED_observed = 2 * np.sum(D_XY) / (n_x * n_y) - np.sum(D_XX) / n_x ** 2 - np.sum(D_YY) / n_y ** 2
    end_time = time.localtime()
    # print("Calculated ED in ", get_time(beg_time,end_time))
    null_hyp_ED_dist = np.zeros(n_permutations)
    if progress:
        permutations_iter = tqdm.tqdm_notebook(range(n_permutations), desc='permutation')
    else:
        permutations_iter = range(n_permutations)
    for i in permutations_iter:
        [i_1, i_2, i_1s, i_2s,
         j_1, j_2, j_1s, j_2s] = permutation_indices(n_x, n_y)

        D_XXs = np.zeros_like(D_XX)
        D_XXs[np.ix_(i_1s, i_1s)] = D_XX[np.ix_(i_1, i_1)]
        D_XXs[np.ix_(i_1s, i_2s)] = D_XY[np.ix_(i_1, i_2)]
        D_XXs[np.ix_(i_2s, i_1s)] = D_YX[np.ix_(i_2, i_1)]
        D_XXs[np.ix_(i_2s, i_2s)] = D_YY[np.ix_(i_2, i_2)]
        D_YYs = np.zeros_like(D_YY)
        D_YYs[np.ix_(j_1s, j_1s)] = D_XX[np.ix_(j_1, j_1)]
        D_YYs[np.ix_(j_1s, j_2s)] = D_XY[np.ix_(j_1, j_2)]
        D_YYs[np.ix_(j_2s, j_1s)] = D_YX[np.ix_(j_2, j_1)]
        D_YYs[np.ix_(j_2s, j_2s)] = D_YY[np.ix_(j_2, j_2)]
        D_XYs = np.zeros_like(D_XY)
        D_XYs[np.ix_(i_1s, j_1s)] = D_XX[np.ix_(i_1, j_1)]
        D_XYs[np.ix_(i_1s, j_2s)] = D_XY[np.ix_(i_1, j_2)]
        D_XYs[np.ix_(i_2s, j_1s)] = D_YX[np.ix_(i_2, j_1)]
        D_XYs[np.ix_(i_2s, j_2s)] = D_YY[np.ix_(i_2, j_2)]
        null_hyp_ED_dist[i] = 2 * np.mean(D_XYs) - np.mean(D_XXs) - np.mean(D_YYs)
    p_value = (n_permutations - np.sum(null_hyp_ED_dist >= ED_observed)) / (n_permutations + 1)
    return p_value, null_hyp_ED_dist, ED_observed


def get_batch_ed_histograms(x, y, batch_size=1000):
    """
    get histograms of energy distance for batches of data with itself (null) and data with generated data (h1)
    :param x: data in the shape [num_samples, num_features]
    :param y: data in the shape [num_samples, num_features]
    :param batch_size:
    :return: null values, H1 values
    """

    for f in x.columns:
        # Just so that any of the features won't take over the metric
        if f in [' rp', ' eneg', ' time']:
            x[f] = np.log(x[f])
            y[f] = np.log(y[f])
        x[f] = (x[f] - np.mean(x[f])) / np.std(x[f])
        y[f] = (y[f] - np.mean(y[f])) / np.std(y[f])

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


def get_batches(array, batch_size):
    batch_list = []
    i = 0
    while i + batch_size < np.shape(array)[0]:
        batch_list.append(array[i:i+batch_size, :])
        i += batch_size
    return batch_list


def get_ed(x, y):
    D_XX = euclidean_distance_matrix(x, x)
    D_XY = euclidean_distance_matrix(x, y)
    D_YY = euclidean_distance_matrix(y, y)
    n_x = np.shape(x)[0]
    n_y = np.shape(y)[0]
    return 2 * np.sum(D_XY) / (n_x * n_y) - np.sum(D_XX) / n_x ** 2 - np.sum(D_YY) / n_y ** 2


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


def make_ed_fig(null, H1, group, show, fig_path):
    fig, axs = plt.subplots(dpi=200)
    axs.grid(True, which='both', color='0.65', linestyle='-')
    bin_max = np.max(np.concatenate((null, H1)))
    bin_min = np.min(np.concatenate((null, H1)))
    bins = np.linspace(bin_min, bin_max, 30)
    axs.hist(null, bins=bins, density=True, alpha=0.6)
    axs.hist(H1, bins=bins, density=True, alpha=0.6)
    ks_test = ks(null, H1)
    axs.set_title(f"{group} ED histogram, ks test p-value:{ks_test.pvalue:.2f}")
    axs.legend(["null", "H1"])
    fig.savefig(fig_path + group + '_histograms.png')


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
    2112: ["Neutron", 0.9395654133, 0, 0.5, 0.5],
    2212: ["Proton", 0.9382720813, +1, 0.5, 0.5],
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
