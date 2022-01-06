from optparse import OptionParser
from sn_fom.utils import transformSN, binned_data
from sn_fom.cosmo_fit import CosmoDist
from sn_fom import plt
import pandas as pd
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter


def sigma_photoz(Om=0.3, w0=-1.0, wa=0., sigma_phot=0.002, plot=False):

    cosmo = CosmoDist()

    zstep = 0.01
    zmin = 0.1
    zmax = 1.2+zstep
    z = np.arange(zmin, zmax, zstep)
    h = 1.e-8
    zh = np.arange(zmin+h, zmax+h, zstep)
    cref = cosmo.mu_astro(z, Om, w0, wa)
    # ctest = cosmo.mu(z, Om, w0, wa)
    ch = cosmo.mu_astro(zh, Om, w0, wa)
    deriv_mu = (ch-cref)/h
    res = pd.DataFrame(z, columns=['z'])
    res['sigma_mu_photoz'] = sigma_phot*(1.+z)*deriv_mu
    res['sigma_photoz'] = sigma_phot
    """
    norm = cosmo.c/cosmo.H0
    norm *= 1.e6
    deriv_dl = cosmo.dL(z, Om, w0, wa)/(1.+z) + \
        cosmo.integrand(z, norm, Om, w0, wa)*(1.+z)
    deriv_mu_true = (5./np.log(10))*deriv_dl/cosmo.dL(z, Om, w0, wa)

    print(deriv_mu/deriv_mu_true, np.log(10))
    """
    if plot:
        fig, ax = plt.subplots()
        ax.plot(res['z'], res['sigma_mu_photoz'])

        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('$\sigma_{photo-z}$')
        plt.show()

    return res


def estimate_syste(data, dbNames, nsigma, plot=False):

    df_syste = pd.DataFrame()

    for dbName in dbNames:
        print('processing', dbName)
        syste = Syste_x1_color(data, dbName=dbName, nsigma=nsigma)
        res = syste()
        df_syste = pd.concat((df_syste, res))

    print('hello z', df_syste['z'].max())
    df_syste.to_hdf(
        'sigma_mu_bias_x1_color_{}_sigma.hdf5'.format(nsigma), key='bias')

    if plot:
        fig, ax = plt.subplots()
        zcomplete = '$z_{complete}$'

        for dbName in df_syste['dbName'].unique():
            idx = df_syste['dbName'] == dbName
            print('cols', df_syste.columns)
            sel_syste = df_syste[idx].to_records(index=False)
            sel_syste.sort(order='z')
            print('ici', dbName, np.max(
                sel_syste['z']), sel_syste[['z', 'delta_mu_bias']])
            zcomp = dbName.split('_')[1]
            ax.plot(sel_syste['z'], sel_syste['delta_mu_bias'],
                    label='{} = {}'.format(zcomplete, zcomp))

        ax.grid()
        ax.legend()
        plt.show()


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
    zmax = 1.2
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


class Syste_x1_color:

    def __init__(self, dd, dbName='DD_0.65', nsigma=1):

        zcomp = dbName.split('_')[1]
        self.data = dd[dbName]
        self.dbName = dbName
        self.zmin = 0.1
        self.zmax = 1.2
        self.zstep = 0.07

        nsigma = str(nsigma)

        self.configs = ['nosigmaInt',
                        'x1_plus_{}_sigma'.format(nsigma),
                        'x1_minus_{}_sigma'.format(nsigma),
                        'color_plus_{}_sigma'.format(nsigma),
                        'color_minus_{}_sigma'.format(nsigma),
                        'sigmaInt_plus_{}_sigma'.format(nsigma),
                        'sigmaInt_minus_{}_sigma'.format(nsigma)]
        """
        corresp = dict(zip(configs, ['nominal',
                                     '$x_1 + {}\sigma$'.format(nsigma),
                                     '$x_1 -{}\sigma$'.format(nsigma),
                                     '$c + {}\sigma$'.format(nsigma),
                                     '$c -{}\sigma$'.format(nsigma)]))
        """

    def __call__(self):

        # get binned data
        df_dict = self.get_binned_data(self.data, self.configs)

        # get diff wrt nominal
        df_syst = self.get_diff_nominal(df_dict, self.configs)

        # finally: get impact on sigma_mu
        df_syst = self.get_bias(df_syst)
        print('finally', df_syst)
        df_syst['dbName'] = self.dbName
        df_syst = df_syst[['dbName', 'z', 'delta_mu_bias']]
        df_syst = df_syst.sort_values(by=['z'])
        last_df = df_syst.iloc[-1]
        first_df = df_syst.iloc[0]
        r = []
        r.append((self.dbName, 1.2, last_df['delta_mu_bias']))
        r.append((self.dbName, 0.01, first_df['delta_mu_bias']))
        dd_app = pd.DataFrame(r, columns=[
            'dbName', 'z', 'delta_mu_bias'])
        df_syst = df_syst.append(dd_app)

        return df_syst

    def get_binned_data(self, data, configs):

        nbins = int(self.zmax/self.zstep)
        bins = np.linspace(self.zmin, self.zmax, nbins)
        df_dict = {}
        vars_fit = ['x1_fit', 'color_fit', 'Mb']
        vars = ['x1', 'color', 'mu']+vars_fit
        corr_cov = dict(zip(vars_fit, ['x1', 'color', 'Mb']))
        for config in configs:
            idx = data['config'] == config
            data_sn = data[idx]
            data_sn = data_sn.rename(columns={"Cov_mbmb": "Cov_MbMb"})
            for vv in ['x1_fit', 'color_fit', 'Mb']:
                data_sn['sigma_{}'.format(vv)] = np.sqrt(
                    data_sn['Cov_{}{}'.format(corr_cov[vv], corr_cov[vv])])
            bdata = pd.DataFrame()
            for vv in vars:
                bdatab = binned_data(self.zmin, self.zmax, nbins, data_sn,
                                     vv, 'sigma_{}'.format(vv))
                if bdata.empty:
                    bdata = bdatab
                else:
                    bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])

            bdata['dbName'] = dbName
            df_dict[config] = bdata

        return df_dict

    def get_diff_nominal(self, df_dict, configs):

        dfsyst = pd.DataFrame()
        ref = df_dict['nosigmaInt']
        ref = ref.replace([np.inf, -np.inf, np.nan], 0)
        ref = ref.round({'z': 6})

        for config in configs[1:]:
            df = df_dict[config]
            df = df.replace([np.inf, -np.inf, np.nan], 0)
            df = df.round({'z': 6})
            if 'plus' in config:

                vvconf = config.split('_')[0]
                if vvconf == 'sigmaInt':
                    vvconf = 'Mb'
                bb = ref.merge(df, left_on=['z'], right_on=['z'])

                for vv in ['x1_fit', 'color_fit', 'Mb']:
                    dvar = ['z']
                    vvb = vv
                    if 'fit' in vv:
                        vvb = vv.split('_')[0]
                    """
                    diff = gaussian_filter(df['{}_fit_mean'.format(
                    vv)], 3)-gaussian_filter(ref['{}_fit_mean'.format(vv)], 3)
                    diff_mb = gaussian_filter(
                    df['Mb_mean'], 3)-gaussian_filter(ref['Mb_mean'], 3)
                    """

                    vvar1 = 'delta_{}_{}'.format(vvb, vvconf)
                    dvar.append(vvar1)
                    bb[vvar1] = bb['{}_mean_x'.format(
                        vv)]-bb['{}_mean_y'.format(vv)]

                    bb = bb.round({'z': 6})

                    if dfsyst.empty:
                        dfsyst = bb[dvar]
                    else:
                        dfsyst = dfsyst.merge(
                            bb[dvar], left_on=['z'], right_on=['z'])

        return dfsyst

    def get_bias(self, dfsyst):

        alpha = 0.13
        beta = 3.1

        # replace deltas by covariance
        dfsyst = dfsyst.rename(columns=lambda x: x.replace('delta', 'Cov'))
        print('hello', dfsyst.columns, dfsyst)
        for vv in ['x1', 'color', 'Mb']:
            vvb = 'Cov_{}_{}'.format(vv, vv)
            dfsyst[vvb] = dfsyst[vvb]**2
            #dfsyst['delta_color'] += dfsyst['delta_color_{}'.format(vv)]
            #dfsyst['delta_Mb'] += dfsyst['delta_Mb_{}'.format(vv)]

        dfsyst['delta_mu_bias'] = alpha**2*dfsyst['Cov_x1_x1']
        dfsyst['delta_mu_bias'] += beta**2*dfsyst['Cov_color_color']
        dfsyst['delta_mu_bias'] += dfsyst['Cov_Mb_Mb']
        # covariance terms
        """
        dfsyst['delta_mu_bias'] += 2.*alpha*dfsyst['Cov_Mb_x1']
        dfsyst['delta_mu_bias'] += -2.*alpha*beta*dfsyst['Cov_color_x1']
        dfsyst['delta_mu_bias'] += -2.*beta*dfsyst['Cov_Mb_color']
        """
        """
        dfsyst['delta_mu_x1'] = dfsyst['delta_Mb_x1']**2 + \
            alpha**2*dfsyst['delta_x1']**2
        dfsyst['delta_mu_color'] = dfsyst['delta_Mb_color']**2 + \
            beta**2*dfsyst['delta_color']**2
        
        dfsyst['delta_mu_x1'] = np.sqrt(alpha**2*dfsyst['delta_x1']**2)
        dfsyst['delta_mu_color'] = np.sqrt(beta**2*dfsyst['delta_color']**2)
        dfsyst['delta_mu_Mb'] = np.sqrt(beta**2*dfsyst['delta_Mb']**2)
        """
        dfsyst['delta_mu_bias'] = np.sqrt(dfsyst['delta_mu_bias'])

        return dfsyst


def plot_test_b(dd, dbName='DD_0.65', nsigma=1):

    zcomp = dbName.split('_')[1]
    data = dd[dbName]
    zmin = 0.1
    zmax = 1.2
    zstep = 0.07

    nbins = int(zmax/zstep)
    bins = np.linspace(zmin, zmax, nbins)

    nsigma = str(nsigma)
    configs = ['nosigmaInt',
               'x1_plus_{}_sigma'.format(nsigma),
               'x1_minus_{}_sigma'.format(nsigma),
               'color_plus_{}_sigma'.format(nsigma),
               'color_minus_{}_sigma'.format(nsigma),
               'sigmaInt_plus_{}_sigma'.format(nsigma),
               'sigmaInt_minus_{}_sigma'.format(nsigma)]

    corresp = dict(zip(configs, ['nominal',
                                 '$x_1 + {}\sigma$'.format(nsigma),
                                 '$x_1 -{}\sigma$'.format(nsigma),
                                 '$c + {}\sigma$'.format(nsigma),
                                 '$c -{}\sigma$'.format(nsigma)]))

    df_dict = {}
    vars_fit = ['x1_fit', 'color_fit', 'Mb']
    vars = ['x1', 'color']+vars_fit
    corr_cov = dict(zip(vars_fit, ['x1', 'color', 'Mb']))
    for config in configs:
        idx = data['config'] == config
        data_sn = data[idx]
        data_sn = data_sn.rename(columns={"Cov_mbmb": "Cov_MbMb"})
        print(config, len(data_sn))
        for vv in ['x1_fit', 'color_fit', 'Mb']:
            data_sn['sigma_{}'.format(vv)] = np.sqrt(
                data_sn['Cov_{}{}'.format(corr_cov[vv], corr_cov[vv])])
        bdata = binned_data(zmin, zmax, nbins, data_sn, 'mu')
        for vv in vars:
            bdatab = binned_data(zmin, zmax, nbins, data_sn,
                                 vv, 'sigma_{}'.format(vv))
            bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])
        # print(test)
        bdata['dbName'] = dbName
        # print(config, bdata[['mu_mean', 'z']])
        df_dict[config] = bdata
        # print('heeeee', bdata[['mu_mean', 'mu_sigma']])
        # print(test)
    # print(test)
    cosmo = CosmoDist()

    fig, ax = plt.subplots()
    ref = df_dict['nosigmaInt']

    for config in configs:
        df = df_dict[config]

        # print(config, df['mu_mean'])
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
        for key, val in enumerate(['x1_fit_mean', 'color_fit_mean', 'Mb_mean']):
            ax[key].plot(df['z'], gaussian_filter(
                df[val], 3), label=corresp[config])

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()
    ax[0].set_ylabel('$<x_1>$')
    ax[1].set_ylabel('$<c>$')
    ax[2].set_ylabel('$<M_b>$')
    ax[2].set_xlabel('$z$')

    """
    fig, ax = plt.subplots(nrows=3)
    fig.suptitle('$z_{complete} $ = '+'{}'.format(zcomp))
    for config in configs:
        df = df_dict[config]
        # print(config, df['mu_mean'])
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
    """
    dfsyst = pd.DataFrame()
    ref = df_dict['nosigmaInt']
    for config in configs[1:]:
        df = df_dict[config]
        for vv in ['x1', 'color']:
            if vv in config and 'plus' in config:
                diff = gaussian_filter(df['{}_fit_mean'.format(
                    vv)], 3)-gaussian_filter(ref['{}_fit_mean'.format(vv)], 3)
                diff_mb = gaussian_filter(
                    df['Mb_mean'], 3)-gaussian_filter(ref['Mb_mean'], 3)
                # diff_sigmu = df['sigma_mu_mean']-ref['sigma_mu_mean']
                # print('here', vv, df['{}_diff'.format(vv)])
                dfg = pd.DataFrame(diff, columns=[
                    'delta_{}'.format(vv)])
                dfg['delta_Mb_{}'.format(vv)] = diff_mb
                dfg['z'] = df['z']
                # dfg['sigma_mu_{}'.format(vv)] = diff_sigmu
                if dfsyst.empty:
                    dfsyst = dfg
                else:
                    dfsyst = dfsyst.merge(dfg, left_on=['z'], right_on=['z'])

    print(dfsyst)
    alpha = 0.13
    beta = 3.1
    dfsyst['delta_mu_x1'] = dfsyst['delta_Mb_x1']+alpha*dfsyst['delta_x1']
    dfsyst['delta_mu_color'] = dfsyst['delta_Mb_color'] - \
        beta*dfsyst['delta_color']
    # dfsyst['delta_mu_x1'] = alpha*dfsyst['delta_x1']
    # dfsyst['delta_mu_color'] = -beta*dfsyst['delta_color']

    dfsyst['delta_mu'] = np.sqrt(
        dfsyst['delta_mu_x1']**2+dfsyst['delta_mu_color']**2)
    """
    dfsyst['sigma_mu'] = np.sqrt(
        dfsyst['sigma_mu_x1']**2+dfsyst['sigma_mu_color']**2)
    """
    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(dfsyst['z'], dfsyst['delta_mu_x1'])
    ax[1].plot(dfsyst['z'], dfsyst['delta_mu_color'])
    ax[2].plot(dfsyst['z'], dfsyst['delta_mu'])
    print('mean systematic on mu', dfsyst['delta_mu'].median())
    plt.show()


def plot_sigma_mu(dd):

    zmin = 0.1
    zmax = 1.2
    nbins = 20

    dres = pd.DataFrame()
    for key, val in dd.items():
        da = binned_data(zmin, zmax, nbins, val)
        da['dbName'] = key
        dres = pd.concat((dres, da))

    fig, ax = plt.subplots(figsize=(12, 9))
    dbNames = ['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65']
    ls = dict(zip(dbNames, ['solid', 'dotted', 'dashed', 'dashdot']))

    for dbName in dbNames:
        idx = dres['dbName'] == dbName
        sel = dres[idx]
        sel = sel.sort_values(by=['z'])
        zcomp = dbName.split('_')[-1]
        ax.plot(sel['z'], sel['sigma_mu_mean'],
                label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)

    ax.grid()
    ax.set_ylabel('<$\sigma_\mu$>')
    ax.set_xlabel('$z$')
    ax.legend()

    plt.show()


def plot_nsn_bias(fileDir, dbNames):

    # special config file needed here: 1 season, 1 pointing per field
    from sn_fom.utils import getconfig
    from sn_fom.steps import NSN_bias

    config = getconfig(['DD_0.90'],
                       ['COSMOS,XMM-LSS,CDFS,ADFS,ELAIS'],
                       ['1,1,1,1,1'],
                       ['1,1,1,1,1'])
    nsn_bias = NSN_bias(fileDir, config,
                        fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                        dbNames=dbNames,
                        plot=True, outName='toplot_bias.hdf5').data


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
    zmax = 1.2
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


def getSN(fileDir, dbName, fitDir, fakes, alpha, beta, Mb, nsigma=1, binned=False):

    outName = 'allSN_{}'.format(dbName)
    if binned:
        outName += '_binned'
    outName += '_{}_sigma'.format(nsigma)
    outName += '.hdf5'

    if not os.path.isfile(outName):
        data = pd.DataFrame()
        for ff in fakes:
            configName = '_'.join(ff.split('_')[1:])
            fDir = '{}/{}/{}'.format(fileDir, ff, fitDir)
            print('fDir', fDir)
            fDir = '{}/{}/{}'.format(fileDir, ff, fitDir)
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
                  default='Fakes_nosigmaInt',
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
parser.add_option("--nsigma", type=int,
                  default=1,
                  help="nsigma for syste estimation [%default]")
parser.add_option("--fitDir", type=str,
                  default='Fit_Ny_40',
                  help="dir with fitted data [%default]")

opts, args = parser.parse_args()

fileDir = opts.fileDir
fakes = opts.fakes.split(',')
fitDir = opts.fitDir
dbNames = opts.dbNames.split(',')
alpha = opts.alpha
beta = opts.beta
Mb = opts.Mb
binned = opts.binned
nsigma = opts.nsigma

# load and transform the data

data = {}

fakes += ['Fakes_x1_plus_{}_sigma'.format(nsigma),
          'Fakes_x1_minus_{}_sigma'.format(nsigma),
          'Fakes_color_plus_{}_sigma'.format(nsigma),
          'Fakes_color_minus_{}_sigma'.format(nsigma),
          'Fakes_sigmaInt_plus_{}_sigma'.format(nsigma),
          'Fakes_sigmaInt_minus_{}_sigma'.format(nsigma)]

fakes += ['Fakes_x1_plus_{}_sigma'.format(nsigma)]

for dbName in dbNames:
    data[dbName] = getSN(fileDir, dbName, fitDir, fakes, alpha,
                         beta, Mb, nsigma=nsigma, binned=0)


# bias syste x1_color
estimate_syste(data, dbNames, nsigma, plot=True)
"""
sigma_phot = 0.02
sigma_photoz = sigma_photoz(plot=False, sigma_phot=sigma_phot)
sigma_photoz.to_hdf('sigma_mu_photoz_{}.hdf5'.format(
    sigma_phot), key='sigma_photoz')
"""
print(test)


if not binned:
    var = 'color'
    """
    plotSN('DD_0.65', 'x1_plus_sigma', data)
    plt.show()
    """
    zrange = 'highz'
    # x1_color_th = plot_x1_color_th(var, zrange=zrange)

    # plot_x1_color(var, 'DD_0.65', data, x1_color_th, zrange=zrange)
    print('hhh', data['DD_0.65']['config'])
    # print(test)
    # plot_x1_color_diff(var, 'DD_0.65', data, zmax=0.1)
    # plot_test_b(data, dbName='DD_0.80', nsigma=nsigma)
    # plot_sigma_mu(data)
    plot_nsn_bias('Fakes_nosigmaInt/Fit', data.keys())
if binned:
    plot_diff('DD_0.65', data)


plt.show()
