from sn_fom.cosmo_fit import zcomp_pixels, FitData, FitData_mu
from sn_fom.utils import loadSN, selSN, update_config, getDist, transformSN, binned_data
from sn_fom.utils import nSN_bin_eff, simu_mu, select
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from . import np


def fit_SN(dbNames, config, fields, snType, sigmu, nsn_bias, sn_wfd=pd.DataFrame(), params_fit=['Om', 'w0', 'wa'], saveSN='', sigma_bias=0.01):

    # simulate supernovae here - DDF
    data_sn = simul_SN(dbNames, config, fields, snType,
                       sigmu, nsn_bias, saveSN=saveSN, sigma_bias=sigma_bias)

    # add WFD SN if required
    print('NSN DD:', len(data_sn), 'WFD:', len(sn_wfd))
    if not sn_wfd.empty:
        data_sn = pd.concat((data_sn, sn_wfd))

    if saveSN != '':
        data_sn.to_hdf(saveSN, key='sn')

    # cosmo fit here
    SNID = ''
    if saveSN != '':
        SNID = saveSN.split('.hdf5')[0]

    # second step: fit the data

    # FitCosmo instance
    fit = FitData_mu(data_sn, params_fit=params_fit)

    # make the fit and get the parameters
    params_fit = fit()

    if SNID != '':
        params_fit['SNID'] = SNID
    return params_fit


def simul_SN(dbNames, config, fields, snType, sigmu, nsn_bias, saveSN='', sigma_bias=0.01):

    data_sn = pd.DataFrame()
    for i, dbName in enumerate(dbNames):
        fields_to_process = fields[i].split(',')
        idx = config['fieldName'].isin(fields_to_process)
        # dd = getSN(fileDir, dbName, config[idx], fields_to_process, snType)
        idxb = sigmu['dbName'] == dbName
        idxa = nsn_bias['zcomp'] == dbName.split('_')[1]
        dd = getSN_mu_simu(dbName, fields_to_process,
                           sigmu[idxb], nsn_bias[idxa], config[idx], sigma_bias)
        data_sn = pd.concat((data_sn, dd))
        print('there', dbName, fields_to_process, dd.size)

    return data_sn


def fit_SN_old(fileDir, dbNames, config, fields, snType, params_fit=['Om', 'w0', 'wa'], sigmu=pd.DataFrame(), saveSN=''):
    data_sn = pd.DataFrame()

    # first step: generate SN
    print('there man', dbNames, sigmu)

    idxb = sigmu['dbName'] == 'WFD_0.20'
    wfd = getSN_mu_simu_wfd(fileDir, 'WFD_0.20', sigmu[idxb])

    print(wfd)
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
    nsn_bias = params['nsn_bias']
    sn_wfd = params['sn_wfd']
    sigma_bias = params['sigma_bias']
    params_fit = pd.DataFrame()
    np.random.seed(123456+j)
    for i in index:
        if saveSN != '':
            saveSN_f = '{}/SN_{}.hdf5'.format(saveSN, i)
        fitpar = fit_SN(dbNames, config,
                        fields, snType,
                        sigma_mu_from_simu, nsn_bias, sn_wfd, params_for_fit, saveSN=saveSN_f, sigma_bias=sigma_bias)
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


def getSN_mu_simu(dbName, fields, sigmu, nsn_bias, config, sigma_bias=0.01):

    zcomp = dbName.split('_')[1]

    SN = pd.DataFrame()
    for field in fields:
        idx = nsn_bias['fieldName'] == field
        nsn_eff = pd.DataFrame(np.copy(nsn_bias[idx].to_records()))
        ia = config['fieldName'] == field
        selconfig = config[ia]
        nseasons = selconfig['nseasons'].to_list()[0]
        nfields = selconfig['nfields'].to_list()[0]
        nsn_eff['nsn_eff'] *= nseasons*nfields
        simuparams = nsn_eff.merge(
            sigmu, left_on=['z'], right_on=['z'])
        # simulate distance modulus (and error) here
        res = simu_mu(simuparams, sigma_bias)
        SN = pd.concat((SN, res))

    print('total number of SN', len(SN))

    SN = SN.droplevel('z')

    return SN


def getSN_mu_simu_old(fileDir, dbName, config, fields, sigmu_from_simu):

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

    SN = SN.droplevel('z')

    return SN


def getSN_mu_simu_wfd(fileDir, dbName, sigmu_from_simu, nfich=-1, nfactor=1):

    zst = np.mean(np.diff(sigmu_from_simu['z']))/2
    zmin = 0.
    zmax = sigmu_from_simu['z'].max()+zst
    bins = np.arange(zmin, zmax, 2.*zst)

    # get SN from simu
    data_sn = select(loadSN(fileDir, dbName, 'WFD', nfich=nfich))

    idx = data_sn['z'] < 0.2
    data_sn = data_sn[idx]
    # get the effective (bias effect) number of expected SN per bin
    group = data_sn.groupby(pd.cut(data_sn.z, bins))
    print(group.size())
    plot_centers = (bins[:-1] + bins[1:])/2
    nsn_eff = pd.DataFrame(plot_centers, columns=['z'])
    nsn_eff['nsn_eff'] = group.size().to_list()
    nsn_eff['nsn_eff'] *= nfactor
    nsn_eff['surveytype'] = 'full'
    nsn_eff['zcomp'] = 0.2
    nsn_eff['zsurvey'] = 0.2

    # print('nsn_eff', nsn_eff)
    # print('sigmu', sigmu_from_simu)
    """
    import matplotlib.pyplot as plt
    plt.hist(data_sn['z'], histtype='step')
    plt.show()
    """

    print('total number of SN', np.sum(nsn_eff['nsn_eff']))
    # merge with sigmu_from_simu
    nsn_eff = nsn_eff.round({'z': 6})
    sigmu_from_simu = sigmu_from_simu.round({'z': 6})
    simuparams = nsn_eff.merge(sigmu_from_simu, left_on=['z'], right_on=['z'])

    # print('after merging', simuparams)
    # simulate distance modulus (and error) here

    SN = simu_mu(simuparams)

    SN = SN.droplevel('z')

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


def SN_WFD(fileDir, sigmu=pd.DataFrame(), saveSN='SN_WFD.hdf5', nfich=-1, nfactor=1):

    import os
    if not os.path.isfile(saveSN):

        # generate SN

        idxb = sigmu['dbName'] == 'WFD_0.20'
        wfd = getSN_mu_simu_wfd(fileDir, 'WFD_0.20',
                                sigmu[idxb], nfich=nfich, nfactor=nfactor)
        wfd.to_hdf(saveSN, key='sn_wfd')

    sn = pd.read_hdf(saveSN)

    return sn


class NSN_bias:
    """
    class to estimate the number of observed SN as a function of the redshift.
    These numbers are estimated from a set of simulated+fitted LCs=SN

    Parameters
    ---------------
    fileDir: str
        location dir of the SN files
    config_fields: list(str), opt
    sigmu: pandas df
      z, sigma_mu, sigma_mu_err for simulated SN
      list of fields to process (default: ['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'])
    dbNames: list(str), opt
      list of configurations to consider (default: ['DD_0.65', 'DD_0.70', 'DD_0.80', 'DD_0.90'])
    plot: bool, opt
     to plot the results (default: False)
    outName: str, opt
      output name (default: nsn_bias.hdf5)


    """

    def __init__(self, fileDir, config,
                 fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                 dbNames=['DD_0.65', 'DD_0.70', 'DD_0.80', 'DD_0.90'],
                 plot=False, outName='nsn_bias.hdf5'):

        self.fileDir = fileDir
        self.config = config
        self.fields = fields
        self.dbNames = dbNames

        import os
        if not os.path.isfile(outName):
            data = self.get_data()
            data.to_hdf(outName, key='nsn')

        self.data = pd.read_hdf(outName)
        if plot:
            self.plot()

    def get_data(self):
        """
         Method to loop on dbNames and fields to estimate the number of sn per redshift bin

         Returns
         ----------
         pandas df with the following columns

         """
        data_sn = pd.DataFrame()

        for i, dbName in enumerate(self.dbNames):
            for field in self.fields:
                idx = self.config['fieldName'].isin([field])
                dd = self.getNSN_bias(dbName, self.config[idx], [field])
                data_sn = pd.concat((data_sn, dd))
                print('SN inter', dbName, len(dd))

        return data_sn

    def getNSN_bias(self, dbName, config, fields):
        """
        Method to estimate the number of SN per redshift bin
        for a configuration and a field

        Parameters
        --------------
        dbName: str
          configuration to consider
        config: dict
          config related to the field: nseason, nfields, season length,...
        fields: list(str)
          fields to process

        Returns
        ----------
        pandas df with the following columns

        """
        zcomp = dbName.split('_')[1]
        zcomplete = float(zcomp)

        config = update_config(fields, config,
                               np.round(zcomplete, 2))

        nsn_scen = NSN_config(config)

        nsn_per_bin = nsn_bin(nsn_scen.data)

        # get SN from simu
        data_sn = loadSN(self.fileDir, dbName, 'allSN')

        # get the effective (bias effect) number of expected SN per bin

        nsn_eff = nSN_bin_eff(data_sn, nsn_per_bin)

        nsn_eff['fieldName'] = fields[0]
        nsn_eff['zcomp'] = zcomp

        return nsn_eff

    def plot(self):
        """
        Method to plot nsn vs z for each field and config

        """
        import matplotlib.pyplot as plt
        for field in self.data['fieldName'].unique():
            idx = self.data['fieldName'] == field
            sel = self.data[idx]
            fig, ax = plt.subplots()
            fig.suptitle('{}'.format(field))
            for zcomp in sel['zcomp'].unique():
                idxb = sel['zcomp'] == zcomp
                selb = sel[idxb]
                ax.plot(selb['z'], selb['nsn_eff'],
                        label='$z_{complete}$'+'= {}'.format(zcomp))

            ax.grid()
            ax.set_xlabel('$z$')
            ax.set_ylabel('N$_{SN}$')
            ax.legend()
        plt.show()
