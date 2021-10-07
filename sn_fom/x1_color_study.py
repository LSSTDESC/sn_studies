from optparse import OptionParser
from sn_fom.utils import transformSN, binned_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plotSN(dbName, config, data, var='Cov_colorcolor'):

    data = data[dbName]
    idx = data['config'] == config
    data = data[idx]
    fig, ax = plt.subplots()
    ax.plot(data['z'], np.sqrt(data[var]), 'ko')


def func(par, par_mean, sigma):
    """
    Function to define the probability of a parameter
    Parameters
    --------------
    par: float
        parameter value
    par_mean: float
        mean value of the parameter
    sigma: float
        sigma of the distribution
    Returns
    ----------
    exp(-(par-par_mean)**2/2/sigma**2
    """
    return np.exp(-(par-par_mean)**2/(2.*sigma**2))


def get_proba_param(fichname='x1_color_G10.csv', sigma_param={}):
    """
    Function to estimate the probability distributions of (x1,c)
    Parameters
    ---------------
    fichname: str
        file with parameters to construct proba distributions
        csv file with the following columns
    zrange,param,param_min,param_max,param_mean,param_mean_err,sigma_minus,sigma_minus_err,sigma_plus,sigma_plus_err
    Probability destribution are estimated from
    Measuring Type Ia Supernova Populations of Stretch and Color and Predicting Distance Biases - D.Scolnic and R.Kessler, The Astrophysical Journal Letters, Volume 822, Issue 2 (2016).
    Returns
    ----------
    pandas df with the following columns
    zrange, param, val, proba
    with
    zrange = lowz/highz
    param = x1/color
    """
    x1_color = pd.read_csv(fichname, comment='#')

    # color distrib

    x1_color_probas = pd.DataFrame()

    for io, row in x1_color.iterrows():
        ppa = np.arange(row.param_min, row.param_mean, 0.001)
        ppb = np.arange(row.param_mean, row.param_max, 0.001)
        pp_all = np.concatenate((ppa, ppb))
        res = pd.DataFrame(pp_all.tolist(), columns=['val'])
        ppval = row.param_mean+sigma_param[row.param]*row.param_mean_err
        # print('here',row.param,ppval,row.param_mean)
        probaa = func(ppa, ppval, row.sigma_minus)
        probab = func(ppb, ppval, row.sigma_plus)
        proba_all = np.concatenate((probaa, probab))
        res['proba'] = proba_all.tolist()
        res['zrange'] = row.zrange
        res['param'] = row.param
        x1_color_probas = pd.concat((x1_color_probas, res))

    return x1_color_probas


def plot_x1_color(var, dbName, dd, x1_color, config='nosigmaInt', zrange='highz', zmin=0.0, zmax=1.2, zstep=0.02):

    corresp = dict(
        zip(['nosigmaInt', '{}_plus_sigma'.format(var), '{}_minus_sigma'.format(var)], ['0', '1', '-1']))

    data = dd[dbName]
    print('hhh', data['config'])
    idx = data['config'] == config
    data = data[idx]

    zmin = data['z'].min()
    zmax = data['z'].max()

    idc = x1_color['nsigma'] == corresp[config]
    sel_x1_color = x1_color[idc]

    nbins = int(zmax/zstep)
    if zrange == 'highz':
        zmin = np.max([0.1, zmin])
    else:
        zmax = np.min([0.1, zmax])

    bins = np.linspace(zmin, zmax, nbins)
    group = data.groupby(pd.cut(data.z, bins))

    for group_name, df_group in group:
        fig, ax = plt.subplots()
        ax.hist(data[var], histtype='step', density=True, bins=50)
        zmin = group_name.left
        zmax = group_name.right
        fig.suptitle('(zmin,zmax) = ({},{})'.format(zmin, zmax))
        ax.hist(sel_x1_color[var], histtype='step',
                bins=50, color='r', density=True)

    plt.show()


def plot_x1_color_th(var='x1', zrange='highz'):

    conf = [0., 3., -3.]
    res = {}
    sig_par = dict(zip(['x1', 'color'], [0.0, 0.0]))
    for v in conf:
        sig_par[var] = v
        res[v] = get_proba_param(
            'reference_files/x1_color_G10.csv', sig_par)
        sig_par[var] = 0.0

    n = 10000
    resout = pd.DataFrame()
    for key, val in res.items():
        idx = val['param'] == var
        idx &= val['zrange'] == zrange
        sel = val[idx]
        norm = np.sum(sel['proba'])
        vv = np.random.choice(sel['val'], n, p=sel['proba']/norm)
        #ax.plot(sel_x1_color[ip]['val'], 10000.*sel_x1_color[ip]['proba'])
        rr = pd.DataFrame(vv, columns=[var])
        rr['nsigma'] = '{}'.format(int(key))
        rr['zrange'] = zrange
        resout = pd.concat((resout, rr))

    colors = dict(zip(['0', '3', '-3'], ['r', 'k', 'k']))
    fig, ax = plt.subplots()
    fig.suptitle(zrange)
    for conf in resout['nsigma'].unique():
        idx = resout['nsigma'] == conf
        vv = resout[idx]
        ax.hist(vv[var], histtype='step', bins=50,
                density=True, color=colors[conf])
    plt.show()
    return resout


def plot_diff(dbName, data):
    """
    Method to plot delta_mu = mu_ref-mu

    Parameters
    --------------
    dbName: str
       config name to plot
    data: pandas df
       data to plot

    """

    dd = data[dbName]

    print(dd['dbName'].unique(), dd['config'].unique())

    fig, ax = plt.subplots()
    fig.suptitle(dbName)
    refconf = 'nosigmaInt'
    idx = dd['config'] == refconf
    ref = dd[idx]
    dictrn = {}
    for vv in ['mu', 'sigma_mu']:
        dictrn['{}_mean'.format(vv)] = '{}_mean_ref'.format(vv)
        dictrn['{}_rms'.format(vv)] = '{}_mean_rms_ref'.format(vv)
    ref = ref.rename(columns=dictrn)
    for cc in dd['config'].unique():
        if cc != refconf:
            idx = dd['config'] == cc
            sel = dd[idx]
            sel = sel.merge(ref, left_on=['z'], right_on=['z'])
            sel['diff_mu'] = (sel['mu_mean_ref'] -
                              sel['mu_mean'])/sel['mu_mean_ref']
            sel = sel.to_records(index=True)
            ax.plot(sel['z'], sel['diff_mu'], label=cc)

    ax.set_xlabel('$z$')
    ax.set_ylabel('$\Delta\mu$')
    ax.grid()
    ax.legend()


def bin_df(data, varlist=['mu', 'sigma_mu']):
    """
    Method to get binned infos from the pandas df

    Parameters
    --------------
    data: pandas df
      data to use for binning

    Returns
    -----------
    pandas df with binned data

    """

    zmin = 0.
    zmax = 1.0
    bin_width = 0.05
    nbins = int(zmax/bin_width)
    dbName = data['dbName'].unique()[0]
    print('dd', dbName)
    config = data['config'].unique()[0]
    print('dd', config)

    dfres = pd.DataFrame()
    for var in varlist:
        df = binned_data(zmin, zmax, nbins, data, var=var)
        if dfres.empty:
            dfres = df
        else:
            dfres = dfres.merge(df, left_on=['z'], right_on=['z'])

    dfres['dbName'] = dbName
    dfres['config'] = config
    return dfres


def getSN(fileDir, dbName, fakes, alpha, beta, Mb, binned=False):

    outName = 'allSN_{}'.format(dbName)
    if binned:
        outName += '_binned'
    outName += '.hdf5'

    if not os.path.isfile(outName):
        data = pd.DataFrame()
        for ff in fakes:
            configName = '_'.join(ff.split('_')[1:])
            fDir = '{}/{}/Fit'.format(fileDir, ff)
            dd = transformSN(fDir, dbName, 'allSN', alpha, beta, Mb)
            dd['dbName'] = dbName
            dd['config'] = configName
            if binned:
                dd = bin_df(dd)
            data = pd.concat((data, dd))
        data.to_hdf(outName, key=dbName)

    data = pd.read_hdf(outName)
    return data


parser = OptionParser(
    description='(x1,color) distribution effects on mu, sigma_mu')

parser.add_option("--fileDir", type="str",
                  default='.',
                  help="main file directory [%default]")
parser.add_option("--fakes", type="str",
                  default='Fakes_nosigmaInt,Fakes_x1_plus_sigma,Fakes_x1_minus_sigma,Fakes_color_plus_sigma,Fakes_color_minus_sigma',
                  help="fake name directory [%default]")
parser.add_option("--dbNames", type="str",
                  default='DD_0.90,DD_0.85,DD_0.80,DD_0.75,DD_0.70,DD_0.65',
                  help="config name [%default]")
parser.add_option("--alpha", type=float,
                  default=0.14,
                  help="alpha parameter for SN standardization [%default]")
parser.add_option("--beta", type=float,
                  default=3.1,
                  help="beta parameter for SN standardization [%default]")
parser.add_option("--Mb", type=float,
                  default=-19.0,
                  help="Mb parameter for SN standardization [%default]")
parser.add_option("--binned", type=int,
                  default=1,
                  help="to get binned data [%default]")
opts, args = parser.parse_args()

fileDir = opts.fileDir
fakes = opts.fakes.split(',')
dbNames = opts.dbNames.split(',')
alpha = opts.alpha
beta = opts.beta
Mb = opts.Mb
binned = opts.binned

# load and transform the data

data = {}
for dbName in dbNames:
    data[dbName] = getSN(fileDir, dbName, fakes, alpha,
                         beta, Mb, binned=binned)

if not binned:
    var = 'x1'
    """
    plotSN('DD_0.65', 'x1_plus_sigma', data)
    plt.show()
    """
    x1_color_th = plot_x1_color_th(var)
    #plot_x1_color(var, 'DD_0.65', data, x1_color_th)
    plot_x1_color(var, 'DD_0.65', data, x1_color_th,
                  config='{}_plus_sigma'.format(var), zmin=1.0)

if binned:
    plot_diff('DD_0.80', data)


plt.show()
