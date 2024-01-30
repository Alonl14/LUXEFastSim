import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generator import (InnerGenerator, OuterGenerator)
import time

def get_kld(real, fake):
    """
    Creates N-dimensional pdfs and calculate their KL-divergence
    :param real: real event features
    :param fake: events generated by the net
    :return: KL-divergence associated with the pdf's
    """

    target_hist = torch.histogram(real.cpu()[:,[0,1]], bins=500, density=True).hist
    current_hist = torch.histogram(fake.cpu()[:,[0,1]], bins=500, density=True).hist
    current_kld = nn.functional.kl_div(nn.functional.log_softmax(target_hist)
                                       , nn.functional.log_softmax(current_hist)
                                       , log_target=True)
    return current_kld


def transform(quantiles, norm, columns, fake_p, dataGroup):
    """
    transform from 'gaussianized' space to actual parameter space
    :param quantiles: The QT object used on the original data
    :param norm: min max normalization data
    :param columns: the required features
    :param fake_p: generated events from the network
    :param dataGroup: specify Inner/Outer
    :return: dataframe of generated events
    """

    temp = quantiles.inverse_transform(fake_p)
    if dataGroup == 'inner':
        temp[:, 0] = (np.copysign(np.abs(temp[:, 0]) ** (9./5), temp[:, 0])) + 0.73
        temp[:, 1] = np.tan(temp[:, 1])/10 + 0.83
        temp[:, [2, 4, 5, 6]] = np.exp(-temp[:, [2, 4, 5, 6]])
        temp[:, 4] = 1 - temp[:, 4]
    else:
        temp[:, [3, 5, 6, 7]] = np.exp(-temp[:, [3, 5, 6, 7]])
        temp[:, 5] = 1 - temp[:, 5]
    df = pd.DataFrame([])

    for i, col in enumerate(columns):
        max = norm['max'][col]
        min = norm['min'][col]
        df[col] = min + (max - min) * temp[:, i]

    df[' phi_x'] = np.arctan2(df[' yy'], df[' xx']) + np.pi
    # if dataGroup == 'outer':
    #     df[' xx'] = df[' rx'] * np.cos(df[' phi_x'] - np.pi)
    #     df[' yy'] = df[' rx'] * np.sin(df[' phi_x'] - np.pi)
    #     df[[' xx', ' yy']] = df[[' xx', ' yy']] + 500
    # el
    if dataGroup == 'inner':
        df[' rx'] = np.sqrt(df[' xx'] ** 2 + df[' yy'] ** 2)

    df[' pxx'] = df[' rp'] * np.cos(df[' phi_p'] - np.pi)
    df[' pyy'] = df[' rp'] * np.sin(df[' phi_p'] - np.pi)
    df['theta'] = np.arccos(df[' pzz'] / np.sqrt(df[' pzz'] ** 2 + df[' rp'] ** 2))
    return df


def plot_correlations(x, y, xlabel, ylabel, bins=[400, 400], loglog=False, Xlim=None, Ylim=None):
    H, xb, yb = np.histogram2d(x, y, bins=bins, range=[[x.min(), x.max()], [y.min(), y.max()]], density=True)
    X, Y = np.meshgrid(xb, yb)
    plt.figure(dpi=250)
    plt.pcolormesh(X, Y, np.log10(H.T))
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
    plt.show()


def make_plots(df, dataGroup):
    """
    Takes a dataframe and its data group and makes correlation plots for x-y,E-t,r_x-theta,phi_x-phi_p
    :param df: dataframe containing both polar and cartesian forms of data
    :param dataGroup: inner/outer
    :return: null
    """
    x_lim = [-4000, 4000]
    y_lim = [-4000, 4000]
    if dataGroup == 'inner':
        x_lim = [-1700, 500]
        y_lim = [-2500, 500]

    plot_correlations(df[' xx'], df[' yy'], 'x[mm]', 'y[mm]', Xlim=x_lim, Ylim=y_lim)
    energy_bins = 10 ** np.linspace(-7, 0, 400)
    time_bins = 10 ** np.linspace(1, 8, 400)
    plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', bins=[time_bins, energy_bins], loglog=True)
    plot_correlations(df[' rx'], df['theta'], 'r [mm]', 'theta_p [rad]')
    plot_correlations(df[' phi_p'], df[' phi_x'], 'phi_p [rad]', 'phi_x [rad]')


def make_plots2(df, dataGroup):
    x_lim = [-1800, 600]
    y_lim = [-2000, 600]
    if dataGroup == 'inner':
        x_lim = [-1700, 500]
        y_lim = [-2500, 500]

    plot_correlations(df[' xx'], df[' yy'], 'x[mm]', 'y[mm]',bins = 800 ,Xlim=x_lim, Ylim=y_lim)
    energy_bins = 10 ** np.linspace(-7, 0, 400)
    time_bins = 10 ** np.linspace(1, 8, 400)
    plot_correlations(df[' time'], df[' eneg'], 't[ns]', 'E[GeV]', bins=[time_bins, energy_bins], loglog=True)
    plot_correlations(df[' rx'], df['theta'], 'r [mm]', 'theta_p [rad]')
    plot_correlations(df[' phi_p'], df[' phi_x'], 'phi_p [rad]', 'phi_x [rad]')


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
        ax2.hist(ds.preqt[:, i], bins=400)
        ax3.hist(ds.data[:, i], bins=400)
        ax4.plot(ds.quantiles.quantiles_[:, i])

        feature = '{:8}'.format(col)

        norm = np.max(get_q(ds)[:, i])
        slope = np.max(get_q(ds)[1:, i] - get_q(ds)[:-1, i]) / norm
        curvature = np.max(get_q(ds)[2:, i] - 2 * get_q(ds)[1:-1, i] + get_q(ds)[:-2, i]) / norm

        print(feature + '%.4f     %.4f' % (slope, curvature))
        # print("\n")

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        plt.show()


def generate_df(trainer, noiseDim, numEvents):
    """
    Create dataframe using net containing numEvents
    :return: generated dataframe
    """
    noise = torch.randn(numEvents, noiseDim, device='cpu')
    generated_data = trainer.genNet(noise)
    generated_data = generated_data.detach().numpy()
    ds = trainer.dataset
    param_list = ds.preprocess.columns
    generated_df = transform(ds.quantiles, ds.norm, param_list,
                             generated_data, trainer.dataGroup)
    return generated_df


def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m,nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)
    elif isinstance(m,nn.InstanceNorm1d):
        nn.init.normal_(m.weight.data, 1, 0.02)


def combine(real_df, innerT, outerT):
    inner, outer = split(real_df)
    numEvents = len(real_df)/25
    q_in = len(inner)/(len(inner)+len(outer))
    inner_events = np.int64(np.floor(numEvents*q_in))
    outer_events = np.int64(np.ceil(numEvents*(1-q_in)))
    print(inner_events, outer_events)
    inner_df = generate_df(innerT, innerT.noiseDim, inner_events)
    outer_df = generate_df(outerT, outerT.noiseDim, outer_events)

    generated_df = pd.concat((inner_df, outer_df), axis=0)

    return generated_df


def check_run(id, real_df, innerTrainer, outerTrainer):
    for var in ["inner", "outer"]:
        cap = var.capitalize()
        exec(var+"Gen = " + cap+"Generator(" + var+"Trainer.noiseDim)")
        exec(var+"""Gen.load_state_dict(torch.load("Output/run_"""+id+"/" + var+"""_Gen_model.pt"
                , map_location=torch.device('cpu')))""")
        exec(var + "Gen = " + cap+"Generator(" + var+"Trainer.noiseDim)")
        exec(var + "Trainer.genNet = " + var+"Gen")
    return combine(real_df, innerTrainer, outerTrainer)


def get_time(end_time, beg_time=np.zeros(9)):
    print(np.int64(end_time) - np.int64(beg_time))
    print(time.asctime(time.struct_time(np.int64(end_time) - np.int64(beg_time))))
    return time.asctime(time.struct_time(np.int64(end_time) - np.int64(beg_time)))[11:19]
