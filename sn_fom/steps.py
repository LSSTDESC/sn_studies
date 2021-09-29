from sn_fom.cosmo_fit import zcomp_pixels, FitData, FitData_mu
from sn_fom.utils import loadSN, selSN, update_config, getDist, transformSN, binned_data
from sn_fom.utils import nSN_bin_eff, simu_mu, select
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from . import np


def fit_SN(fileDir, dbNames, config, fields, snType, params_fit=['Om', 'w0', 'wa'], sigmu=pd.DataFrame(), saveSN=''):
    data_sn = pd.DataFrame()

    # first step: generate SN
    print('there man', dbNames, sigmu)

    idxb = sigmu['dbName'] == 'WFD_0.20'
    wfd = getSN_mu_simu_wfd(fileDir, 'WFD_0.20', sigmu[idxb])
    print(test)
    for i, dbName in enumerate(dbNames):
        fields_to_process = fields[i].split(',')
        idx = config['fieldName'].isin(fields_to_process)
        # dd = getSN(fileDir, dbName, config[idx], fields_to_process, snType)
        idxb = sigmu['dbName'] == dbName
        dd = getSN_mu_simu(
            fileDir, dbName, config[idx], fields_to_process, sigmu[idxb])
        data_sn = pd.concat((data_sn, dd))
        print('SN inter', dbName, len(dd))

    data_sn = data_sn.droplevel('z')

    SNID = ''
    if saveSN != '':
        SNID = saveSN.split('.hdf5')[0]
        data_sn.to_hdf(saveSN, key='sn')

    # second step: fit the data

    # FitCosmo instance
    fit = FitData_mu(data_sn, params_fit=params_fit)

    # make the fit and get the parameters
    params_fit = fit()

    if SNID != '':
        params_fit['SNID'] = SNID
    return params_fit


def multifit(index, params, j=0, output_q=None):

    fileDir = params['fileDir']
    dbNames = params['dbNames']
    config = params['config']
    fields = params['fields']
    saveSN = params['dirSN']
    snType = params['snType']
    sigma_mu_from_simu = params['sigma_mu']
    params_for_fit = params['params_fit']

    params_fit = pd.DataFrame()
    np.random.seed(123456+j)
    for i in index:
        if saveSN != '':
            saveSN_f = '{}/SN_{}.hdf5'.format(saveSN, i)
        fitpar = fit_SN(fileDir, dbNames, config,
                        fields, snType,
                        params_for_fit, sigma_mu_from_simu, saveSN=saveSN_f)
        params_fit = pd.concat((params_fit, fitpar))

    if output_q is not None:
        return output_q.put({j: params_fit})
    else:
        return params_fit


def getSN(fileDir, dbName, config, fields, snType):

    tt = zcomp_pixels(fileDir, dbName, 'faintSN')
    zcomp = tt()
    print('redshift completeness', zcomp)
    zcomplete = zcomp['zcomp'][0]
    # zcomplete = 1.
    config = update_config(fields, config,
                           np.round(zcomplete, 2))
    print('config updated', config)

    # get number of supernovae
    nsn_scen = NSN_config(config)
    print('total number of SN', nsn_scen.nsn_tot())
    nsn_per_bin = nsn_bin(nsn_scen.data)
    print(nsn_per_bin, nsn_per_bin['nsn_survey'].sum())

    # get SN from simu
    data_sn = loadSN(fileDir, dbName, snType, zcomp)

    # load x1_c distrib
    x1_color = getDist()

    # select according to nsn_per_bin
    data_sn = selSN(data_sn, nsn_per_bin, x1_color)

    print(len(data_sn), type(data_sn))

    return data_sn


def getSN_mu_simu(fileDir, dbName, config, fields, sigmu_from_simu):

    zcomp = dbName.split('_')[1]
    zcomplete = float(zcomp)

    config = update_config(fields, config,
                           np.round(zcomplete, 2))

    nsn_scen = NSN_config(config)

    nsn_per_bin = nsn_bin(nsn_scen.data)

    # get SN from simu
    data_sn = loadSN(fileDir, dbName, 'allSN')

    # get the effective (bias effect) number of expected SN per bin

    nsn_eff = nSN_bin_eff(data_sn, nsn_per_bin)

    # merge with sigmu_from_simu

    simuparams = nsn_eff.merge(sigmu_from_simu, left_on=['z'], right_on=['z'])

    # simulate distance modulus (and error) here

    SN = simu_mu(simuparams)
    return SN


def getSN_mu_simu_wfd(fileDir, dbName, sigmu_from_simu):
    print(sigmu_from_simu, np.mean(np.diff(sigmu_from_simu['z'])))
    zst = np.mean(np.diff(sigmu_from_simu['z']))/2
    zmin = 0.
    zmax = sigmu_from_simu['z'].max()+zst
    bins = np.arange(zmin, zmax, 2.*zst)

    # get SN from simu
    data_sn = select(loadSN(fileDir, dbName, 'WFD'))

    idx = data_sn['z'] < 0.2
    data_sn = data_sn[idx]
    # get the effective (bias effect) number of expected SN per bin
    group = data_sn.groupby(pd.cut(data_sn.z, bins))
    print(group.size())
    plot_centers = (bins[:-1] + bins[1:])/2
    nsn_eff = pd.DataFrame(plot_centers, columns=['z'])
    nsn_eff['nsn_eff'] = group.size().to_list()
    nsn_eff['surveytype'] = 'full'
    nsn_eff['zcomp'] = 0.2
    nsn_eff['zsurvey'] = 0.2

    print(nsn_eff)
    """
    import matplotlib.pyplot as plt
    plt.hist(data_sn['z'], histtype='step')
    plt.show()
    """

    # merge with sigmu_from_simu

    simuparams = nsn_eff.merge(sigmu_from_simu, left_on=['z'], right_on=['z'])

    print(simuparams)
    # simulate distance modulus (and error) here

    SN = simu_mu(simuparams)
    return SN


class Sigma_mu_obs:
    """
    class to estimate the error on the distance modulus vs z
    used fully simulated+fitted light curves

    """

    def __init__(self, fileDir,
                 dbNames=['DD_0.90', 'DD_0.65',
                          'DD_0.70', 'DD_0.80', 'WFD_0.20'],
                 snTypes=['allSN']*4+['WFD'],
                 outName='sigma_mu_from_simu.hdf5',
                 plot=False,
                 alpha=0.13, beta=3.1):

        self.fileDir = fileDir
        self.dbNames = dbNames
        self.snTypes = snTypes
        self.outName = outName
        self.alpha = alpha
        self.beta = beta

        import os
        if not os.path.isfile(outName):
            res = self.sigma_mu()
            res.to_hdf(outName, key='sigmamu')

        self.data = pd.read_hdf(outName)
        if plot:
            self.plot(self.data)

    def sigma_mu(self):
        """
        Method to estimate sigma_mu for SN

        """

        zmin = 0.0

        df = pd.DataFrame()
        for io, dbName in enumerate(self.dbNames):
            zmax = 1.0
            nbins = 21
            if 'WFD' in dbName:
                zmax = 0.20
                nbins = 20

            snType = self.snTypes[io]
            data_sn = transformSN(self.fileDir, dbName,
                                  snType, self.alpha, self.beta)
            bdata = binned_data(zmin, zmax, nbins, data_sn)
            bdata['dbName'] = dbName
            df = pd.concat((df, bdata))
        return df

    def plot(self, df):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for dbName in df['dbName'].unique():
            idx = df['dbName'] == dbName
            bdata = df[idx]
            ax.errorbar(bdata['z'], bdata['sigma_mu_mean'],
                        yerr=bdata['sigma_mu_rms'], label=dbName)

        ax.legend()
        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('$\sigma_\mu$ [mag]')
        plt.show()
