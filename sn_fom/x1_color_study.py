from optparse import OptionParser
from sn_fom.utils import transformSN, binned_data
from sn_fom.cosmo_fit import CosmoDist
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


def plot_test(data):

    zmin = 0.01
    zmax = 1.0
    zstep = 0.02

    nbins = int(zmax/zstep)
    bins = np.linspace(zmin, zmax, nbins)

    df = pd.DataFrame()
    config = 'x1_plus_2_sigma'
    config = 'nosigmaInt'
    for dbName in data.keys():
        idx = data[dbName]['config'] == config
        data_sn = data[dbName][idx]

        bdata = binned_data(zmin, zmax, nbins, data_sn)
        bdatab = binned_data(zmin, zmax, nbins, data_sn, 'mu')
        bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])
        bdata['dbName'] = dbName
        df = pd.concat((df, bdata))

    print(df.columns)
    fig, ax = plt.subplots()
    idx = df['dbName'] == 'DD_0.90'
    ref = df[idx]
    cosmo = CosmoDist()
    mu_th = cosmo.mu_astro(ref['z'], 0.3, -1.0, 0.0)
    df_th = pd.DataFrame(mu_th, columns=['mu_th'])
    df_th['z'] = ref['z']

    for dbName in df['dbName'].unique():
        idx = df['dbName'] == dbName
        sel = df[idx]
        sel = sel.merge(df_th, left_on=['z'], right_on=['z'])
        sel['diff_mu'] = sel['mu_th']-sel['mu_mean']
        ax.plot(sel['z'], sel['diff_mu'], label=dbName, ls='None', marker='o')

    ax.grid()
    ax.legend()
    plt.show()


def plot_test_b(dd, dbName='DD_0.65'):

    zcomp = dbName.split('_')[1]
    data = dd[dbName]
    zmin = 0.1
    zmax = 1.0
    zstep = 0.05

    nbins = int(zmax/zstep)
    bins = np.linspace(zmin, zmax, nbins)

    configs = ['nosigmaInt',
               'x1_plus_1_sigma', 'x1_minus_1_sigma',
               'color_plus_1_sigma', 'color_minus_1_sigma']
    corresp = dict(zip(configs, ['nominal', '$x_1 + 1\sigma$', '$x_1 -1\sigma$',
                                 '$c + 1\sigma$', '$c -1\sigma$']))

    df_dict = {}
    for config in configs:
        idx = data['config'] == config
        data_sn = data[idx]
        print(config, data_sn.columns)
        bdata = binned_data(zmin, zmax, nbins, data_sn, 'mu')
        for vv in ['x1', 'x1_fit', 'color', 'color_fit', 'Mb', 'Cov_x1mb', 'Cov_colormb', 'Cov_x1color']:
            bdatab = binned_data(zmin, zmax, nbins, data_sn, vv)
            bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])

        bdata['dbName'] = dbName
        #print(config, bdata[['mu_mean', 'z']])
        df_dict[config] = bdata
        print('heeeee', bdata[['mu_mean', 'mu_sigma']])
        print(test)

    cosmo = CosmoDist()

    fig, ax = plt.subplots()
    ref = df_dict['nosigmaInt']
    for config in configs:
        df = df_dict[config]
        #print(config, df['mu_mean'])
        mu_th = cosmo.mu_astro(df['z'], 0.3, -1.0, 0.0)
        df_th = pd.DataFrame(mu_th, columns=['mu_th'])
        df_th['z'] = df['z']
        df = df.merge(df_th, left_on=['z'], right_on=['z'])
        df['diff_mu'] = df['mu_th']-df['mu_mean']
        ax.plot(df['z'], ref['mu_mean']-df['mu_mean'], label=config)

    fig, ax = plt.subplots(nrows=3)
    fig.suptitle('$z_{complete} $ = '+'{}'.format(zcomp))
    for config in configs:
        df = df_dict[config]
        #print(config, df['mu_mean'])
        print(config, df[['z', 'x1_fit_mean', 'color_fit_mean']])
        ax[0].plot(df['z'], df['x1_fit_mean'], label=corresp[config])
        ax[1].plot(df['z'], df['color_fit_mean'], label=corresp[config])
        ax[2].plot(df['z'], df['Mb_mean'], label=corresp[config])

    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[0].set_ylabel('$<x_1>$')
    ax[1].set_ylabel('$<c>$')
    ax[2].set_ylabel('$<M_b>$')
    ax[2].set_xlabel('$z$')

    fig, ax = plt.subplots(nrows=3)
    fig.suptitle('$z_{complete} $ = '+'{}'.format(zcomp))
    for config in configs:
        df = df_dict[config]
        #print(config, df['mu_mean'])
        ax[0].plot(df['z'], df['Cov_x1mb_mean'], label=corresp[config])
        ax[1].plot(df['z'], df['Cov_colormb_mean'], label=corresp[config])
        ax[2].plot(df['z'], df['Cov_x1color_mean'], label=corresp[config])

    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[0].set_ylabel('$<x_1>$')
    ax[1].set_ylabel('$<c>$')
    ax[2].set_ylabel('$<M_b>$')
    ax[2].set_xlabel('$z$')

    dfsyst = pd.DataFrame()
    ref = df_dict['nosigmaInt']
    for config in configs[1:]:
        df = df_dict[config]
        for vv in ['x1', 'color']:
            if vv in config and 'plus' in config:
                print(config)
                diff = df['{}_fit_mean'.format(
                    vv)]-ref['{}_fit_mean'.format(vv)]
                diff_mb = df['Mb_mean']-ref['Mb_mean']
                diff_sigmu = df['sigma_mu_mean']-ref['sigma_mu_mean']
                #print('here', vv, df['{}_diff'.format(vv)])
                dfg = pd.DataFrame(diff.values, columns=[
                                   'sigma_{}'.format(vv)])
                dfg['sigma_Mb_{}'.format(vv)] = diff_mb
                dfg['z'] = df['z']
                dfg['sigma_mu_{}'.format(vv)] = diff_sigmu
                if dfsyst.empty:
                    dfsyst = dfg
                else:
                    dfsyst = dfsyst.merge(dfg, left_on=['z'], right_on=['z'])

    print(dfsyst)
    alpha = 0.13
    beta = 3.1
    dfsyst['sigma_mu'] = dfsyst['sigma_Mb_x1']**2 + dfsyst['sigma_Mb_color']**2 + \
        alpha**2*dfsyst['sigma_x1']**2+beta**2*dfsyst['sigma_color']**2
    dfsyst['sigma_mu'] = np.sqrt(dfsyst['sigma_mu'])
    dfsyst['sigma_mu'] = np.sqrt(
        dfsyst['sigma_mu_x1']**2+dfsyst['sigma_mu_color']**2)
    fig, ax = plt.subplots()
    ax.plot(dfsyst['z'], dfsyst['sigma_mu'])
    plt.show()


def plot_x1_color(var, dbName, dd, x1_color, config='nosigmaInt', zrange='highz', zmin=0.0, zmax=1.2, zstep=0.02):

    corresp = dict(
        zip(['nosigmaInt', '{}_plus_2_sigma'.format(var), '{}_minus_2_sigma'.format(var)], ['0', '2', '-2']))

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

    ig = -1
    for group_name, df_group in group:
        ig += 1
        fig, ax = plt.subplots()
        ax.hist(df_group[var], histtype='step', density=True, bins=50)
        zmin = group_name.left
        zmax = group_name.right
        fig.suptitle('(zmin,zmax) = ({},{})'.format(zmin, zmax))
        ax.hist(sel_x1_color[var], histtype='step',
                bins=50, color='r', density=True)
        plt.savefig('test_plots/{}_{}.png'.format(var, ig))
    plt.show()


def plot_x1_color_diff(var, dbName, dd, zrange='highz', zmin=0.0, zmax=1.2, zstep=0.02):

    configs = ['nosigmaInt', '{}_plus_2_sigma'.format(
        var), '{}_minus_2_sigma'.format(var)]

    colors = dict(zip(configs, ['r', 'k', 'k']))
    data = dd[dbName]
    print('hhh', data['config'])

    zmin = data['z'].min()
    zmax = data['z'].max()

    sel_data = {}
    for config in configs:
        idx = data['config'] == config
        sel_data[config] = data[idx]

    nbins = int(zmax/zstep)
    if zrange == 'highz':
        zmin = np.max([0.1, zmin])
    else:
        zmax = np.min([0.1, zmax])

    bins = np.linspace(zmin, zmax, nbins)

    groups = {}
    for key, vals in sel_data.items():
        groups[key] = vals.groupby(pd.cut(vals.z, bins))

    for group_name, df_group in groups[configs[0]]:
        fig, ax = plt.subplots()
        ax.hist(df_group[var], histtype='step', density=True,
                bins=50, color=colors[configs[0]])
        grpa = groups[configs[1]].get_group(group_name)
        grpb = groups[configs[2]].get_group(group_name)
        zmin = group_name.left
        zmax = group_name.right
        fig.suptitle('(zmin,zmax) = ({},{})'.format(zmin, zmax))
        ax.hist(grpa[var], histtype='step', density=True,
                bins=50, color=colors[configs[1]])
        ax.hist(grpb[var], histtype='step', density=True,
                bins=50, color=colors[configs[2]])
    plt.show()


def plot_x1_color_th(var='x1', zrange='highz'):

    conf = [0., 2., -2.]
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
        # ax.plot(sel_x1_color[ip]['val'], 10000.*sel_x1_color[ip]['proba'])
        rr = pd.DataFrame(vv, columns=[var])
        rr['nsigma'] = '{}'.format(int(key))
        rr['zrange'] = zrange
        resout = pd.concat((resout, rr))

    colors = dict(zip(['0', '2', '-2'], ['r', 'k', 'k']))
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

    print('aoaoao', dd['dbName'].unique(), dd['config'].unique())

    refconf = 'nosigmaInt'
    idx = dd['config'] == refconf
    ref = dd[idx]
    dictrn = {}
    for vv in ['mu', 'sigma_mu']:
        dictrn['{}_mean'.format(vv)] = '{}_mean_ref'.format(vv)
        dictrn['{}_rms'.format(vv)] = '{}_mean_rms_ref'.format(vv)
    ref = ref.rename(columns=dictrn)

    cosmo = CosmoDist()
    mu_th = cosmo.mu_astro(ref['z'], 0.3, -1.0, 0.0)
    df_th = pd.DataFrame(mu_th, columns=['mu_th'])
    df_th['z'] = ref['z']
    fig, ax = plt.subplots()
    ref = ref.merge(df_th, left_on=['z'], right_on=['z'])
    ref['diff_mu'] = ref['mu_th']-ref['mu_mean_ref']
    ax.plot(ref['z'], ref['diff_mu'])
    ax.grid()

    fig, ax = plt.subplots()
    fig.suptitle(dbName)
    for cc in dd['config'].unique():
        if cc != refconf:
            idx = dd['config'] == cc
            sel = dd[idx]
            sel = sel.merge(ref, left_on=['z'], right_on=['z'])
            sel['diff_mu'] = (sel['mu_mean_ref'] -
                              sel['mu_mean'])
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
    outName += '_1_sigma'
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
                  default='Fakes_nosigmaInt,Fakes_x1_plus_1_sigma,Fakes_x1_minus_1_sigma,Fakes_color_plus_1_sigma,Fakes_color_minus_1_sigma',
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
                  default=-19.074,
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
    var = 'color'
    """
    plotSN('DD_0.65', 'x1_plus_sigma', data)
    plt.show()
    """
    zrange = 'highz'
    #x1_color_th = plot_x1_color_th(var, zrange=zrange)

    #plot_x1_color(var, 'DD_0.65', data, x1_color_th, zrange=zrange)
    print('hhh', data['DD_0.65']['config'])
    #plot_x1_color_diff(var, 'DD_0.65', data, zmax=0.1)
    plot_test_b(data, dbName='DD_0.90')


if binned:
    plot_diff('DD_0.65', data)


plt.show()
