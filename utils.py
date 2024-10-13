import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import importlib

import dataset
import generator
from generator import (InnerGenerator, OuterGenerator)

# from Archive.pre7generator import OuterGenerator (InnerGenerator, )
importlib.reload(generator)
import time
from scipy.stats import wasserstein_distance as wd
from scipy.stats import kstest, chisquare
import json
import os
import tqdm


def get_kld(real, fake):
    """
    Creates N-dimensional pdfs and calculate their KL-divergence
    :param real: real event features
    :param fake: events generated by the net
    :return: KL-divergence associated with the pdf's
    """

    target_hist = torch.histogram(real.cpu(), bins=200, density=True).hist
    current_hist = torch.histogram(fake.cpu(), bins=200, density=True).hist
    current_kld = nn.functional.kl_div(nn.functional.log_softmax(target_hist)
                                       , nn.functional.log_softmax(current_hist)
                                       , log_target=True)
    return current_kld


def transform(norm, columns, fake_p, dataGroup, quantiles=None):
    """
    transform from 'gaussianized' space to actual parameter space
    :param quantiles: The QT object used on the original data
    :param norm: min max normalization data
    :param columns: the required features
    :param fake_p: generated events from the network
    :param dataGroup: specify Inner/Outer
    :return: dataframe of generated events
    """
    if quantiles is not None:
        temp = quantiles.inverse_transform(fake_p)
    else:
        temp = np.maximum(-10 * np.ones_like(fake_p), fake_p)
        print("no QT")
    if dataGroup == 'inner':
        # temp[:, 0] = (np.copysign(np.abs(temp[:, 0]) ** (9. / 5), temp[:, 0])) + 0.73
        # temp[:, 1] = np.tan(temp[:, 1]) / 10 + 0.83
        # temp[:, 4] = 1 - temp[:, 4]
        temp[:, [2, 6, 7, 8]] = np.exp(-temp[:, [2, 6, 7, 8]])
        temp[:, 6] = 1 - temp[:, 6]
    else:
        temp[:, [3, 5, 8, 9]] = np.exp(-temp[:, [3, 5, 8, 9]])  # if pre26 , uncomment
        # if pre15 add , 6 and tab these 2 lines
        # else:  # for pre15 versions
        #     temp[:, [5, 7, 8, 9]] = np.exp(-temp[:, [5, 7, 8, 9]])
        #     temp[:, 7] = 1 - temp[:, 7]
        #     # temp[:, [3, 5, 6, 7]] = np.exp(-temp[:, [3, 5, 6, 7]])  # if pre26 , uncomment
        temp[:, 5] = 1 - temp[:, 5]
    df = pd.DataFrame([])

    for i, col in enumerate(columns):
        x_max = norm['max'][col]
        x_min = norm['min'][col]
        df[col] = dataset.normalize(df[col], x_min, x_max, inverse=True)

    # if dataGroup == 'outer':
    #     df[' xx'] = df[' rx'] * np.cos(df[' phi_x'] - np.pi)
    #     df[' yy'] = df[' rx'] * np.sin(df[' phi_x'] - np.pi)
    #     df[[' xx', ' yy']] = df[[' xx', ' yy']] + 500
    # el
    # if dataGroup == 'inner':

    m_neutron = 0.9395654133

    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)
    df[' pzz'] = -np.sqrt((df[' eneg'] + m_neutron) ** 2 - m_neutron ** 2 - df[' rp'] ** 2)
    # df[' eneg'] = np.sqrt(df[' rp']**2+df[' pzz']**2+m_neutron**2)-m_neutron
    # df[' rp'] = np.sqrt((df[' eneg']+m_neutron)**2-m_neutron**2-df[' pzz']**2)
    # df[' rp'] = np.sqrt(df[' eneg']**2-df[' pzz']**2)
    # df[' phi_p'] = np.arctan2(df[' pyy'], df[' pxx'])+np.pi
    df[' pxx'] = df[' rp'] * np.cos(df[' phi_p'] - np.pi)
    df[' pyy'] = df[' rp'] * np.sin(df[' phi_p'] - np.pi)
    df['theta'] = np.arccos(df[' pzz'] / np.sqrt((df[' eneg'] + m_neutron) ** 2 - m_neutron ** 2))
    return df


def plot_correlations(x, y, xlabel, ylabel, run_id, key,
                      bins=[400, 400], loglog=False, Xlim=None, Ylim=None, xData=None, yData=None):
    if xData is None and yData is None:
        H, xb, yb = np.histogram2d(x, y, bins=bins, range=[[x.min(), x.max()], [y.min(), y.max()]], density=True)
        X, Y = np.meshgrid(xb, yb)
        plt.figure(dpi=250)
        plt.pcolormesh(X, Y, H.T, norm="log")
    else:
        if type(bins[0]) == int:
            bins = [np.linspace(xData.min(), xData.max(), 401), np.linspace(yData.min(), yData.max(), 401)]
        H1, xb1, yb1 = np.histogram2d(xData, yData, bins=bins,
                                      range=[[xData.min(), xData.max()], [yData.min(), yData.max()]], density=True)
        H2, xb2, yb2 = np.histogram2d(x, y, bins=bins, range=[[xData.min(), xData.max()], [yData.min(), yData.max()]],
                                      density=True)
        X, Y = np.meshgrid(xb2, yb2)
        plt.figure(dpi=250)
        COM = ((H2 - H1) ** 2 / (H1 + 0.0000000001)).T
        plt.pcolormesh(X, Y, COM, norm='log')
        chi2 = 0
        for i in range(400):
            for j in range(400):
                binX = np.diff(xb1)
                binY = np.diff(yb1)
                chi2 += COM[i, j] * binX[i] * binY[j]
        print(chi2)

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
    if xData is None:
        if run_id is not None:
            path = 'Output/run_' + run_id + '/plots/2dHists'
            if not os.path.isdir(path + '/' + key):
                if not os.path.isdir(path):
                    os.mkdir(path)
                os.mkdir(path + '/' + key)
            plt.savefig(path + '/' + key + '/' + xlabel + '-' + ylabel + '.png')
    plt.show()
    if xData is None:
        return H


def make_plots(df, dataGroup, run_id=None, key=None):
    """
    Takes a dataframe and its data group and makes correlation plots for x-y,E-t,r_x-theta,phi_x-phi_p
    :param df: dataframe containing both polar and cartesian forms of data
    :param dataGroup: inner/outer
    :param run_id: string
    :return: null
    """
    # BEAM TRIM
    x_lim = [-4500, 1500]
    y_lim = [-3000, 6000]
    # # R CUT
    # x_lim = [-4000, 4000]
    # y_lim = [-4000, 4000]
    if dataGroup == 'inner':
        x_lim = [-1700, 500]
        y_lim = [-2500, 500]

    Hxy = plot_correlations(df[' xx'], df[' yy'], 'x[mm]', 'y[mm]', run_id, key, Xlim=x_lim, Ylim=y_lim)
    energy_bins = 10 ** np.linspace(-12, 0, 400)
    time_bins = 10 ** np.linspace(1, 8, 400)
    Het = plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', run_id, key, bins=[time_bins, energy_bins],
                            loglog=True)
    Hrth = plot_correlations(df[' rx'], df['theta'], 'r [mm]', 'theta_p [rad]', run_id, key)
    Hpp = plot_correlations(df[' phi_p'], df[' phi_x'], 'phi_p [rad]', 'phi_x [rad]', run_id, key)
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
    df['theta'] = df['theta'] = np.arccos(df[' pzz'] / np.sqrt(df[' pzz'] ** 2 + df[' rp'] ** 2))


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
        # ax2.hist(ds.preqt[:, i], bins=400)
        ax3.hist(ds.data[:, i], bins=400)
        # ax4.plot(ds.quantiles.quantiles_[:, i])

        feature = '{:8}'.format(col)

        # norm = np.max(get_q(ds)[:, i])
        # slope = np.max(get_q(ds)[1:, i] - get_q(ds)[:-1, i]) / norm
        # curvature = np.max(get_q(ds)[2:, i] - 2 * get_q(ds)[1:-1, i] + get_q(ds)[:-2, i]) / norm
        #
        # print(feature + '%.4f     %.4f' % (slope, curvature))
        # print("\n")

        ax1.set_yscale('log')
        # ax2.set_yscale('log')
        ax3.set_yscale('log')
        plt.show()


def generate_df(trainer, noiseDim, numEvents):
    """
    Create dataframe using net containing numEvents
    :return: generated dataframe
    """
    noise = torch.randn(numEvents, noiseDim, device='cpu')
    trainer.genNet.to('cpu')
    generated_data = trainer.genNet(noise)
    generated_data = generated_data.detach().numpy()
    ds = trainer.dataset
    param_list = ds.preprocess.columns
    try:
        generated_df = transform(ds.norm, param_list, generated_data, trainer.dataGroup, quantiles=ds.quantiles)
    except AttributeError:
        print("No quantiles found in ds, assuming applyQT=0")
        generated_df = transform(ds.norm, param_list, generated_data, trainer.dataGroup)
    return generated_df


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


def generate_trained_df(run_id, trainer):
    """
    Given a run id load the trained model in run_id dir to its respective trainer and return the generated dataframe
    :param run_id: run_id
    :param trainer
    :return:
    """
    if trainer.dataGroup == "outer":
        generator = nn.DataParallel(OuterGenerator(trainer.noiseDim))
    else:
        generator = nn.DataParallel(InnerGenerator(trainer.noiseDim))
    path = "Output/run_" + run_id + "/" + trainer.dataGroup + "_Gen_model.pt"

    generator.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    trainer.genNet = generator
    factor = 10
    return generate_df(trainer, trainer.noiseDim, np.int64(len(trainer.dataset.data) / factor))


def check_run(run_id, innerData, outerData,
              innerTrainer=None, outerTrainer=None):
    run_dir = 'Output/run_' + run_id + '/'
    fig_path = run_dir + 'plots'
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
    fig_path += '/'

    if (innerTrainer is not None) and (outerTrainer is not None):
        innerDF = generate_trained_df(run_id, innerTrainer)
        outerDF = generate_trained_df(run_id, outerTrainer)
    else:
        innerDF = pd.read_csv(run_dir + 'innerDF.csv')
        outerDF = pd.read_csv(run_dir + 'outerDF.csv')

    iKLPath = run_dir + "KL_in.npy"
    oKLPath = run_dir + "KL_out.npy"
    iDLPath = run_dir + "D_Losses_in.npy"
    oDLPath = run_dir + "D_Losses_out.npy"
    innerKLDiv = np.load(iKLPath)
    outerKLDiv = np.load(oKLPath)
    innerDLosses = np.load(iDLPath)
    outerDLosses = np.load(oDLPath)

    plt.figure(dpi=200)
    plt.title("Inner KL divergence")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(innerKLDiv)
    plt.yscale('log')
    plt.savefig(fig_path + 'innerKLDiv.png')
    plt.figure(dpi=200)
    plt.title("Outer KL divergence")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outerKLDiv)
    plt.yscale('log')
    plt.savefig(fig_path + 'outerKLDiv.png')
    plt.figure(dpi=200)
    plt.title("Inner Discriminator Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(innerDLosses)
    plt.savefig(fig_path + 'innerDLosses.png')
    plt.figure(dpi=200)
    plt.title("Outer Discriminator Losses")
    plt.grid(True, which='both', color='0.65', linestyle='-')
    plt.plot(outerDLosses)
    plt.savefig(fig_path + 'outerDLosses.png')
    plt.show()

    features = [' xx', ' yy', ' pxx', ' pyy', ' pzz', ' eneg', ' time', 'theta']
    chi2_tests = {'inner': {}, 'outer': {}, 'combined': {}, 'noLeaks': {}}
    dfDict = {'inner': {}, 'outer': {}, 'combined': {}, 'noLeaks': {}}
    chi2_inner = {' xx': 0, ' yy': 0, ' pxx': 0, ' pyy': 0,
                  ' pzz': 0, ' eneg': 0, ' time': 0, 'theta': 0}
    chi2_outer = chi2_inner.copy()
    chi2_combined = chi2_inner.copy()
    chi2_noLeaks = chi2_inner.copy()
    innerData, outerData = innerData[features], outerData[features]
    combinedData = pd.concat([innerData, outerData])
    combinedDF = pd.concat([innerDF, outerDF])
    noLeaksData = combinedData.copy()
    posIn = ((innerDF[' xx'] < 500) * (innerDF[' xx'] > -1700) * (innerDF[' yy'] < 500))
    posOut = ((outerDF[' xx'] < 500) * (outerDF[' xx'] > -1700) * (outerDF[' yy'] < 500))
    noLeakInner = innerDF[posIn]
    noLeakOuter = outerDF[~posOut]
    noLeaksDF = pd.concat([noLeakInner, noLeakOuter])
    dfDict['inner'] = innerDF
    dfDict['outer'] = outerDF
    dfDict['combined'] = combinedDF
    dfDict['noLeaks'] = noLeaksDF

    Hxy, Het, Hrth, Hpp = make_plots(innerDF, "inner", run_id, 'inner')
    Hxy, Het, Hrth, Hpp = make_plots(outerDF, "outer", run_id, 'outer')
    Hxy, Het, Hrth, Hpp = make_plots(noLeaksDF, "outer", run_id, 'noLeaks')
    GHxy, GHet, GHrth, GHpp = make_plots(combinedDF, "outer", run_id, 'combined')

    make_polar_features(combinedData)

    plot_correlations(combinedDF[' xx'], combinedDF[' yy'], 'x[mm]', 'y[mm]', run_id, key="combined"
                      , Xlim=[-4500, 1500], Ylim=[-3000, 6000], xData=combinedData[' xx'], yData=combinedData[' yy'])
    energy_bins = 10 ** np.linspace(-12, 0, 401)
    time_bins = 10 ** np.linspace(1, 8, 401)
    plot_correlations(combinedDF[' time'], combinedDF[' eneg'], 't[ns]', 'E[GeV]', run_id, key="combined",
                      bins=[time_bins, energy_bins], loglog=True, xData=combinedData[' time'],
                      yData=combinedData[' eneg'])
    plot_correlations(combinedDF[' rx'], combinedDF['theta'], 'r [mm]', 'theta_p [rad]', run_id, key="combined",
                      xData=combinedData[' rx'], yData=combinedData['theta'])
    plot_correlations(combinedDF[' phi_p'], combinedDF[' phi_x'], 'phi_p [rad]', 'phi_x [rad]', run_id, key="combined",
                      xData=combinedData[' phi_p'], yData=combinedData[' phi_x'])

    # for key in chi2_tests.keys():
    #     if not os.path.isdir(fig_path+'1dHists/'+key):
    #         if not os.path.isdir(fig_path+'1dHists'):
    #             os.mkdir(fig_path+'1dHists')
    #         os.mkdir(fig_path+'1dHists/'+key)
    #     for feat in features:
    #         exec("chi2_"+key+"[feat] = kstest("+key+"DF[feat],"+key+"Data[feat]).pvalue")
    #         exec("print(chi2_"+key+"[feat])")
    #         exec("plot_1d("+key+"Data,"+key+"DF,feat,chi2_"+key+", fig_path, key)")
    #     exec("chi2_tests['"+key+"']=chi2_"+key)
    # chi_obj = json.dumps(chi2_tests, indent=8)
    # with open(run_dir + "chi2_tests.json", "w") as outfile:
    #     outfile.write(chi_obj)

    return chi2_tests, dfDict


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


import warnings


# takes log of energy and time, normalizes everything
def prep_matrix(x):
    x[[2, 4, 5], :] = np.log(x[[2, 4, 5], :])

    for d in range(np.shape(x)[0]):
        x[d, :] = x[d, :] - np.min(x[d, :]) / (np.max(x[d, :]) - np.min(x[d, :]))
    return x


# calculates the Euclidean distance matrix for two np arrays x and y
def euclidean_distance_matrix(x, y):
    D_XY = np.zeros([np.shape(x)[1], np.shape(y)[1]])
    shape = np.shape(D_XY)
    for i in range(shape[0]):
        for j in range(shape[1]):
            D_XY[i, j] = np.linalg.norm(x[:, i] - y[:, j])
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
