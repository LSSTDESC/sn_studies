from sn_fom.cosmo_fit import zcomp_pixels, FitData, FitData_mu
from sn_fom.utils import loadSN, selSN, update_config, getDist, transformSN, binned_data
from sn_fom.utils import nSN_bin_eff, simu_mu, select
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from scipy.interpolate import interp1d
from . import np


class fit_SN_mu:

    def __init__(self, fileDir, dbNames, config, fields,
                 snType, sigmu, nsn_bias,
                 sn_wfd=pd.DataFrame(),
                 params_fit=['Om', 'w0', 'wa'],
                 saveSN='', sigmaInt=0.12, sigma_bias=0.01, binned_cosmology=False):

        self.fileDir = fileDir
        self.dbNames = dbNames
        self.config = config
        self.fields = fields
        self.sigmu = sigmu
        self.snType = snType
        self.nsn_bias = nsn_bias
        self.sn_wfd = sn_wfd
        self.params_fir = params_fit
        self.saveSN = saveSN
        self.sigmaInt = sigmaInt
        self.sigma_bias = sigma_bias

        # simulate supernovae here - DDF
        data_sn = self.simul_SN()

        # add bias corr and resimulate
        data_sn = self.add_sigbias_resimulate(data_sn, sigmaInt)
        """
        print('after bias corr', data_sn)
        import matplotlib.pyplot as plt
        plt.plot(data_sn['z_SN'], data_sn['sigma_bias_stat'], 'ko')
        plt.show()
        """
        #data_sn['sigma_bias_stat'] = 0.0
        # add WFD SN if required
        # print('NSN DD:', len(data_sn), 'WFD:', len(sn_wfd))
        data_sn['snType'] = 'DD'
        if not sn_wfd.empty:
            sn_wfd['snType'] = 'WFD'
            sn_wfd['dbName'] = 'WFD'
            sn_wfd['sigma_bias_stat'] = 0.0
            data_sn = pd.concat((data_sn, sn_wfd))

        # print(test)

        if binned_cosmology:
            data_sn = self.binned_SN(data_sn, sigmaInt)

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
        # print('no binning', params_fit)
        """
        # try a binned cosmology here
        binned_sn = self.binned_SN(data_sn)
        print('binned sn', binned_sn)
        # FitCosmo instance
        fit = FitData_mu(binned_sn, params_fit=params_fit)
        # make the fit and get the parameters
        params_fit = fit()
        print('binned', params_fit)
        """
        # print(test)
        if SNID != '':
            params_fit['SNID'] = SNID

        self.params_fit = params_fit

    def binned_SN(self, data, sigmaInt=0.12, zmin=0.0, zmax=1.0, nbins=20):

        zmin = 0.0
        zmax = 1.
        """
        nbins = 168
        bins = np.linspace(zmin, zmax, nbins)
        """
        bins = [0.01, 0.2]
        # bins += np.arange(0.2, 0.2, 0.01).tolist()
        bins += np.arange(0.2, 0.8-0.01, 0.01).tolist()
        bins += np.arange(0.8, zmax+0.005, 0.005).tolist()
        bins = np.array(np.unique(bins))
        # rint(len(bins))
        # bins = np.array(bins_lowz+bin_bins_highz)

        group = data.groupby(pd.cut(data.z_SN, bins))
        # print(group.size())
        bin_centers = (bins[:-1] + bins[1:])/2
        # bin_values = group['mu_SN'].mean().to_list()

        bin_values = group.apply(
            lambda x: np.sum(x['mu_SN']/(x['sigma_mu_SN']**2+sigmaInt**2)) /
            (np.sum(1./(x['sigma_mu_SN']**2+sigmaInt**2))))

        error_values = group.apply(
            lambda x: np.sum(1./x['sigma_mu_SN']**2))

        df = pd.DataFrame(bin_centers, columns=['z_SN'])
        df['mu_SN'] = bin_values.values
        df['sigma_mu_SN'] = 1./np.sqrt(error_values.values)
        df['sigma_bias'] = group['sigma_bias'].mean().to_list()
        df['snType'] = 'DD'
        # print(df)
        # print(test)
        idx = df['z_SN'] <= 0.2
        df.loc[idx, 'snType'] = 'WFD'

        return df

    def simul_SN(self):

        data_sn = pd.DataFrame()
        for i, dbName in enumerate(self.dbNames):
            fields_to_process = self.fields[i].split(',')
            idx = self.config['fieldName'].isin(fields_to_process)
            # dd = getSN(fileDir, dbName, config[idx], fields_to_process, snType)
            idxb = self.sigmu['dbName'] == dbName
            idxa = self.nsn_bias['zcomp'] == dbName.split('_')[1]
            dd = self.getSN_mu_simu(dbName, fields_to_process,
                                    self.sigmu[idxb], self.nsn_bias[idxa], self.config[idx])
            dd['dbName'] = dbName
            data_sn = pd.concat((data_sn, dd))
            # print('there', dbName, fields_to_process, dd.size)
        # print(test)
        return data_sn

    def getSN_mu_simu(self, dbName, fields, sigmu, nsn_bias, config):

        zcomp = dbName.split('_')[1]

        SN = pd.DataFrame()
        # sn_simu = transformSN(self.fileDir, dbName, 'allSN',
        #                      alpha=0.13, beta=3.1, Mb=-19.)
        for field in fields:
            idx = nsn_bias['fieldName'] == field
            nsn_eff = pd.DataFrame(np.copy(nsn_bias[idx].to_records()))
            ia = config['fieldName'] == field
            selconfig = config[ia]
            nseasons = selconfig['nseasons'].to_list()[0]
            nfields = selconfig['nfields'].to_list()[0]
            nsn_eff['nsn_eff'] *= nseasons*nfields
            # sigmu = self.get_sigmu_from_simu(sn_simu, nsn_eff)
            simuparams = nsn_eff.merge(
                sigmu, left_on=['z'], right_on=['z'])
            # simulate distance modulus (and error) here
            # print(field, 'nsn to simulate', np.sum(
            #    simuparams['nsn_eff']), nseasons, nfields)
            res = simu_mu(simuparams, self.sigmaInt, self.sigma_bias)
            # print(field, 'nsn simulated', len(res))
            SN = pd.concat((SN, res))

        # print('total number of SN', len(SN))

        SN = SN.droplevel('z')

        return SN

    def get_sigmu_from_simu(self, sn_simu, nsn_bias):

        zstep = np.mean(nsn_bias['z'].diff())
        nbins = len(nsn_bias)
        zmin = nsn_bias['z'].min()-zstep/2
        zmax = nsn_bias['z'].max()+zstep/2
        bins = np.linspace(zmin, zmax, nbins+1)
        bin_centers = (bins[:-1] + bins[1:])/2
        group = sn_simu.groupby(pd.cut(sn_simu.z, bins))
        r = []

        for group_name, df_group in group:
            zgmin = group_name.left
            zgmax = group_name.right
            zgmean = 0.5*(zgmin+zgmax)
            idf = np.abs(nsn_bias['z']-zgmean) < 1.e-8
            nsn_for_simu = int(nsn_bias[idf]['nsn_eff'])
            # grab randomly some sn here
            if nsn_for_simu < 1:
                nsn_for_simu = 1
            df_sample = df_group.sample(n=nsn_for_simu)
            """
            import matplotlib.pyplot as plt
            plt.hist(df_sample['x1_fit'], histtype='step')
            plt.show()
            """
            sigma_mu = np.sqrt(1./np.sum(df_sample['sigma_mu']**2))
            r.append((zgmean, sigma_mu))

        res = pd.DataFrame(r, columns=['z', 'sigma_mu_mean'])
        res['sigma_mu_rms'] = 0.
        return res

    def bias_stat_error(self, data):
        """
        Method to estimate the statistical uncertainty due to the Malmquist bias

        Parameters
        --------------
        data: pandas df
          data to process

        Returns
        ----------
        original pandas df plus the stat uncertainty due to the Malmquist bias

        """

        df_tot = pd.DataFrame()
        for dbName in data['dbName'].unique():
            idx = data['dbName'] == dbName
            sel = data[idx]
            if 'DD' in dbName:
                idxs = self.sigmu['dbName'] == dbName
                sel_sigmu = self.sigmu[idxs]
                mu_interp = interp1d(
                    sel_sigmu['z'], sel_sigmu['mu_mean'], bounds_error=False, fill_value=0.)
                sigmu_interp = interp1d(
                    sel_sigmu['z'], sel_sigmu['sigma_mu_mean'], bounds_error=False, fill_value=0.)
                # get the redshift completeness value
                zcomp = float(dbName.split('_')[-1])
                # consider only SN with z>zcomp
                idxb = sel['z_SN'] >= zcomp
                selSN = sel[idxb]
                # get the number of SN per bin
                zmin = zcomp
                zmax = 1.1
                zstep = 0.05
                bins = np.arange(zmin, zmax+zstep, zstep)
                io = bins <= zmax
                bins = bins[io]

                grouped = selSN.groupby(pd.cut(selSN.z_SN, bins))
                dd = pd.DataFrame()
                for name, group in grouped:
                    zmean = np.mean([name.left, name.right])
                    group['sigma_bias_stat'] = group['mu_SN']*sigmu_interp(
                        zmean)/mu_interp(zmean)
                    group['sigma_bias_stat'] /= np.sqrt(len(group))
                    dd = pd.concat((dd, group))
                # selSN['size'] = selSN.groupby(
                #    pd.cut(selSN.z_SN, bins)).transform('size')
                df_tot = pd.concat((df_tot, dd))
                df_tot = pd.concat((df_tot, sel[~idxb]))
            else:
                df_tot = pd.concat((df_tot, sel))

        #print('after', df_tot)
        return df_tot

    def add_sigbias_resimulate(self, data_sn, sigmaInt):

        # add a column sigma_bias_stat for all

        data_sn['sigma_bias_stat'] = 0.0
        data_sn = self.bias_stat_error(data_sn)

        # re-estimate the distance moduli including the bias error - for DD only - faster
        from sn_fom.cosmo_fit import CosmoDist
        from random import gauss
        cosmo = CosmoDist(H0=70.)
        dist_mu = cosmo.mu_astro(data_sn['z_SN'], 0.3, -1.0, 0.0)
        sigmu = np.array(data_sn['sigma_mu_SN'])
        sigbias = np.array(data_sn['sigma_bias_stat'])
        #print('hello', len(dist_mu), len(sigmu), sigbias)
        data_sn['mu_SN'] = [gauss(dist_mu[i], np.sqrt(sigmu[i]**2+sigmaInt**2+sigbias[i]**2))
                            for i in range(len(dist_mu))]
        return data_sn


def fit_SN_deprecated(dbNames, config, fields, snType, sigmu, nsn_bias, sn_wfd=pd.DataFrame(), params_fit=['Om', 'w0', 'wa'], saveSN='', sigma_bias=0.01):

    # simulate supernovae here - DDF
    data_sn = simul_SN(dbNames, config, fields, snType,
                       sigmu, nsn_bias, saveSN=saveSN, sigma_bias=sigma_bias)

    # add WFD SN if required
    print('NSN DD:', len(data_sn), 'WFD:', len(sn_wfd))
    data_sn['snType'] = 'DD'
    if not sn_wfd.empty:
        sn_wfd['snType'] = 'WFD'
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


def simul_SN_deprecated(dbNames, config, fields, snType, sigmu, nsn_bias, saveSN='', sigma_bias=0.01):

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


def multifit_mu(index, params, j=0, output_q=None):

    fileDir = params['fileDir']
    dbNames = params['dbNames']
    config = params['config']
    fields = params['fields']
    snDir = params['dirSN']
    snType = params['snType']
    sigma_mu_from_simu = params['sigma_mu']
    params_for_fit = params['params_fit']
    nsn_bias = params['nsn_bias']
    sn_wfd = params['sn_wfd']
    sigma_bias = params['sigma_bias']
    sigmaInt = params['sigmaInt']
    binned_cosmology = params['binned_cosmology']

    params_fit = pd.DataFrame()
    np.random.seed(123456+j)
    for i in index:
        if snDir != '':
            saveSN_f = '{}/SN_{}.hdf5'.format(snDir, i)
        else:
            saveSN_f = ''

        fitpar = fit_SN_mu(fileDir, dbNames, config,
                           fields, snType, sigma_mu_from_simu,
                           nsn_bias, sn_wfd, params_for_fit,
                           saveSN=saveSN_f, sigmaInt=sigmaInt,
                           sigma_bias=sigma_bias, binned_cosmology=binned_cosmology)
        params_fit = pd.concat((params_fit, fitpar.params_fit))

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


def getSN_mu_simu_deprecated(dbName, fields, sigmu, nsn_bias, config, sigma_bias=0.01):

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

    simuparams = nsn_eff.merge(sigmu_from_simu, left_on=[
        'z'], right_on=['z'])

    # simulate distance modulus (and error) here

    SN = simu_mu(simuparams)

    SN = SN.droplevel('z')

    return SN


def getSN_mu_simu_wfd(fileDir, dbName, sigmu_from_simu, nfich=-1, nsn=5000, sigmaInt=0.12):

    zst = np.mean(np.diff(sigmu_from_simu['z']))/2
    print('alors man', zst)
    zmin = 0.
    zmax = sigmu_from_simu['z'].max()+zst
    bins = np.arange(zmin, zmax, 2.*zst)

    # get SN from simu
    data_sn = select(loadSN(fileDir, dbName, 'WFD', nfich=nfich))

    idx = data_sn['z'] < 0.2
    data_sn = data_sn[idx]
    # get the effective (bias effect) number of expected SN per bin
    group = data_sn.groupby(pd.cut(data_sn.z, bins))
    nsn_simu = len(data_sn)
    plot_centers = (bins[:-1] + bins[1:])/2
    nsn_eff = pd.DataFrame(plot_centers, columns=['z'])
    nsn_eff['nsn_eff'] = group.size().to_list()
    nsn_eff['nsn_eff'] *= nsn/nsn_simu
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

    # print('total number of SN to simulate', np.sum(nsn_eff['nsn_eff']))
    # merge with sigmu_from_simu
    nsn_eff = nsn_eff.round({'z': 6})
    sigmu_from_simu = sigmu_from_simu.round({'z': 6})
    simuparams = nsn_eff.merge(sigmu_from_simu, left_on=[
        'z'], right_on=['z'])

    # print(simuparams)
    # print('after merging', simuparams)
    # simulate distance modulus (and error) here

    SN = simu_mu(simuparams, sigmaInt=0.12)

    # print('after simu', len(SN))

    SN = SN.droplevel('z')

    return SN


class Sigma_mu_obs:
    """
    class to estimate the error on the distance modulus vs z
    used fully simulated+fitted light curves

    """

    def __init__(self, fileDir,
                 dbNames=['DD_0.65', 'DD_0.70',
                          'DD_0.75', 'DD_0.80',
                          'DD_0.85', 'DD_0.90', 'WFD_0.20'],
                 snTypes=['allSN']*6+['WFD'],
                 outName='sigma_mu_from_simu.hdf5',
                 plot=False,
                 alpha=0.13, beta=3.1, Mb=-19.0):

        self.fileDir = fileDir
        self.dbNames = dbNames
        self.snTypes = snTypes
        self.outName = outName
        self.alpha = alpha
        self.beta = beta
        self.Mb = Mb

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
                                  snType, self.alpha, self.beta, self.Mb)
            bdata = binned_data(zmin, zmax, nbins, data_sn)
            bdatab = binned_data(zmin, zmax, nbins, data_sn, 'mu')
            bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])
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
                        yerr=bdata['sigma_mu_sigma'], label=dbName)

        ax.legend()
        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('$\sigma_\mu$ [mag]')
        plt.show()


def SN_WFD(fileDir, sigmu=pd.DataFrame(), saveSN='SN_WFD.hdf5', nfich=-1, nsn=5000, sigmaInt=0.12):

    import os
    if not os.path.isfile(saveSN):

        # generate SN

        idxb = sigmu['dbName'] == 'WFD_0.20'
        wfd = getSN_mu_simu_wfd(fileDir, 'WFD_0.20',
                                sigmu[idxb], nfich=nfich, nsn=nsn, sigmaInt=sigmaInt)
        wfd.to_hdf(saveSN, key='sn_wfd')

    sn = pd.read_hdf(saveSN)

    return sn


class NSN_bias:
    """
    class to estimate the number of observed SN as a function of the redshift.
    These numbers are estimated from a set of simulated+fitted LCs = SN

    Parameters
    ---------------
    fileDir: str
        location dir of the SN files
    config_fields: list(str), opt
    sigmu: pandas df
      z, sigma_mu, sigma_mu_err for simulated SN
      list of fields to process(default: ['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'])
    dbNames: list(str), opt
      list of configurations to consider(default: ['DD_0.65', 'DD_0.70', 'DD_0.80', 'DD_0.90'])
    plot: bool, opt
     to plot the results(default: False)
    outName: str, opt
      output name(default: nsn_bias.hdf5)

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
            print('processing', dbName)
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
        print('loading SN')
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
        from scipy.interpolate import make_interp_spline

        for field in self.data['fieldName'].unique():
            idx = self.data['fieldName'] == field
            sel = self.data[idx]
            fig, ax = plt.subplots()
            zcomps = sel['zcomp'].unique()
            zcomps = ['0.90', '0.80', '0.70', '0.65']
            ls = dict(
                zip(zcomps, ['solid', 'dotted', 'dashed', 'dashdot']))
            fig.suptitle('{}'.format(field))
            for zcomp in zcomps:
                idxb = sel['zcomp'] == zcomp
                selb = sel[idxb].to_records(index=False)
                # ax.plot(selb['z'], selb['nsn_eff'],
                #        label='$z_{complee}$'+'= {}'.format(zcomp))
                xnew = np.linspace(
                    np.min(selb['z']), np.max(selb['z']), 100)
                spl = make_interp_spline(
                    selb['z'], selb['nsn_eff'], k=3)  # type: BSpline
                spl_smooth = spl(xnew)
                ax.plot(xnew, spl_smooth,
                        label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[zcomp], lw=3)
            ax.grid()
            ax.set_xlabel('$z$')
            ax.set_ylabel('N$_{\mathrm{SN}}$')
            ax.legend()
            ax.set_xlim([0.05, None])
            ax.set_ylim([0.0, None])
        plt.show()
