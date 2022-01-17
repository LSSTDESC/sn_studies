from sn_fom.cosmo_fit import zcomp_pixels, FitData, FitData_mu
from sn_fom.utils import loadSN, selSN, update_config, getDist, transformSN, binned_data
from sn_fom.utils import nSN_bin_eff, simu_mu, select
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from scipy.interpolate import interp1d
from . import np


class fit_SN_mu:
    """
    class to simulate distance modulus and fit cosmology

    Parameters
    ---------------
    fileDir: str
      directory files
    dbNames: list(str)
      list of dbNames
    config: dict
      survey configuration
    fields: list(str)
      list of fields to consider
    snType: str
      type of SN
    sigmu: pandas df
      sigma_mu values (vs z)
    nsn_bias: pandas df
      nsn_bias values vs z
    sn_wfd: pandas df
      data for WFD
    params_fit: list(str)
      list of cosmology parameters to fit
    saveSN: str, opt
      dir where to save produced SN (default: '')
    sigmaInt: float, opt
      intrinsic dispersion of SN (default: 0.12)
    binned_cosmology: bool, opt
      to perform binned cosmology (not operational yet)(default: False)
    surveyType: str, opt
      type of survey (default: full)
    sigma_mu_photoz: pandas df, opt
      mu error from photoz (default: empty df)
    nsn_WFD_yearly: int,opt
      number of SN in the WFD per year (default: -1)
     nsn_WFD_hostz: int, opt
       number of SN in the WFD with z-spectro host (default: 100000)
     nsn_WFD_hostz_yearly: int, opt
       number of SN in the WFD with z-spectro host per year (default: 500)
     year_survey: int, opt
       current year of the survey (default: 10)
    """

    def __init__(self, fileDir, dbNames, config, fields,
                 snType, sigmu, nsn_bias,
                 sn_wfd=pd.DataFrame(),
                 params_fit=['Om', 'w0', 'wa'],
                 saveSN='', sigmaInt=0.12,
                 sigma_bias_x1_color=pd.DataFrame(),
                 binned_cosmology=False, surveyType='full',
                 sigma_mu_photoz=pd.DataFrame(),
                 nsn_WFD_yearly=-1,
                 nsn_WFD_hostz=100000,
                 nsn_WFD_hostz_yearly=500,
                 year_survey=10):
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
        self.sigma_bias_x1_color = sigma_bias_x1_color
        self.sigma_mu_photoz = sigma_mu_photoz

        self.interp_field = {}
        if not self.sigma_mu_photoz.empty:
            self.info_photoz(nsn_WFD_yearly)

        # simulate supernovae here - DDF
        data_sn = self.simul_distmod()
        # add bias corr and resimulate
        data_sn = self.add_sigbias_resimulate(data_sn, sigmaInt)

        """
        print('after bias corr', data_sn)
        import matplotlib.pyplot as plt
        plt.plot(data_sn['z_SN'], data_sn['sigma_bias_stat'], 'ko')
        plt.show()
        """
        # data_sn['sigma_bias_stat'] = 0.0
        # add WFD SN if required
        # print('NSN DD:', len(data_sn), 'WFD:', len(sn_wfd))
        data_sn['snType'] = 'DD'
        if not sn_wfd.empty:
            sn_wfd = self.get_wfd(sn_wfd, nsn_WFD_yearly,
                                  nsn_WFD_hostz, nsn_WFD_hostz_yearly, sigmaInt, year_survey)
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
        print('to fit', data_sn.columns)
        import matplotlib.pyplot as plt
        for dbName in data_sn['dbName'].unique():
            idx = data_sn['dbName'] == dbName
            sel = data_sn[idx]
            sel = sel.sort_values(by=['z_SN'])
            plt.plot(sel['z_SN'], sel['sigma_bias_x1_color'], label=dbName)

        plt.legend()
        plt.show()

        # FitCosmo instance
        fit = FitData_mu(data_sn, params_fit=params_fit, surveyType=surveyType)

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

    def get_wfd(self, sn_wfd_orig, nsn_WFD_yearly, nsn_WFD_hostz, nsn_WFD_hostz_yearly, sigmaInt, year_survey):
        """
        Method to build the WFD sample

        Parameters
        --------------
        sn_wfd: pandas df
           original WFD data
        nsn_WFD_yearly: int
           number of WFD SN per year
        nsn_WFD_hostz: int
           number of WFD SN with host spectro (total)
        nsn_WFD_hostz_yearly: int
           number of WFD SN with host spectro (per year)
        sigmaInt: float
          SN internal dispersion
        year_survey: int
          year of the survey

        Returns
        ----------
         pandas df of the WFD sample

        """
        sn_wfd = pd.DataFrame(sn_wfd_orig)
        sn_wfd['snType'] = 'WFD'
        sn_wfd['dbName'] = 'WFD'
        sn_wfd['fieldName'] = 'WFD'
        sn_wfd['sigma_bias_stat'] = 0.0
        sn_wfd['sigma_bias_x1_color'] = 0.0
        sn_wfd['sigma_mu_photoz'] = 0.0
        sn_wfd['zcomp'] = 0.6
        sn_wfd['dd_type'] = 'unknown'
        n_wfd_hostz = nsn_WFD_yearly
        if nsn_WFD_yearly > 0:
            #nseasons = self.get_nseasons(fieldList=self.config['fieldName'])
            n_wfd = nsn_WFD_yearly * year_survey
            n_wfd_hostz = nsn_WFD_hostz_yearly * year_survey

            if n_wfd <= len(sn_wfd):
                sn_wfd = sn_wfd.sample(n=n_wfd)
        if not self.sigma_mu_photoz.empty:
            sn_wfd = self.apply_photoz_wfd(
                sn_wfd, nsn_WFD_hostz, n_wfd_hostz, sigmaInt)

        return sn_wfd

    def apply_photoz_wfd(self, sn_wfd, nsn_WFD_hostz, n_wfd_hostz, sigmaInt):
        """
        Method to apply photoz error on the wfd sample

        Parameters
        ---------------
        sn_wfd: pandas df
          data to process
        nsn_WFD_hostz: int
          number of sn with spectro-z (total)
        n_wfd_hostz: int
          number of sn with spectro-z
        sigmaInt: float
          SN internal dispersion

        Returns
        ----------
        pandas df with photoz error propagated on the distance modulus

        """

        res = pd.DataFrame()
        nsn_wfd = len(sn_wfd)

        n_wfd_hostz_ref = np.min([nsn_WFD_hostz, n_wfd_hostz])

        if n_wfd_hostz <= 0.:
            n_wfd_hostz_ref = nsn_WFD_hostz

        if nsn_wfd <= n_wfd_hostz_ref:
            return sn_wfd
        else:
            sn_wfd = sn_wfd.reset_index()
            nsn_noz = nsn_wfd-n_wfd_hostz_ref

            sn_wfd_noz = sn_wfd.sample(n=nsn_noz)
            res = sn_wfd.drop(sn_wfd_noz.index)

            # apply photometric error on mu
            sn_wfd_noz['sigma_mu_photoz'] = self.sigmu_interp(
                sn_wfd_noz['z_SN'])

            # recompute mus
            sn_wfd_noz['mu_SN'] = self.simu_mu_SN(sn_wfd_noz, sigmaInt)

            # rebuild the full pandas df
            sn_wfd_noz = sn_wfd_noz.reset_index()
            res = pd.concat((res, sn_wfd_noz))

        # check here
        """
        idx = res['sigma_mu_photoz'] > 1.e-5
        print('WFD without host z', len(res[idx]))
        print('WFD with host z-spectro', len(res)-len(res[idx]))
        """
        return res

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
        bin_centers = (bins[: -1] + bins[1:])/2
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

    def simul_distmod(self):
        """
        Method to simulate distance moduli
        loop on the configurations (dbNames) and  DD fields.
        (1 field = 1 dbName)

        Returns
        ----------
        pandas df of distance moduli

        """
        data_sn = pd.DataFrame()
        # loop on configurations
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
        """
        Get distance moduli

        Parameters
        ---------------
        dbName: str
          name of the config (DD_*)
        fields: list(str)
          list of fields to process
        sigmu: pandas df
          array of distance modulus errors (with Malmquist bias) vs z
        nsn_bias: pandas df
          array of NSN (with Malmquist bias) vs z
        config: dict
          survey configuration
        """
        zcomp = float(dbName.split('_')[1])

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
            # print('aaaaa', sigmu, nsn_eff)
            # print('aoooaooo', sigmu['z'], nsn_eff['z'])

            sigmu = sigmu.round({'z': 3})
            nsn_eff = nsn_eff.round({'z': 3})
            simuparams = nsn_eff.merge(
                sigmu, left_on=['z'], right_on=['z'])
            # print('there man after merge', simuparams)
            # simulate distance modulus (and error) here
            # print(field, 'nsn to simulate', np.sum(
            #    simuparams['nsn_eff']), nseasons, nfields)

            ido = simuparams['z'] <= 1.1
            simuparams = simuparams[ido]
            res = simu_mu(simuparams, self.sigmaInt)

            res['fieldName'] = field
            res['zcomp'] = zcomp
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

    def add_bias_stat(self, data):
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
                """
                zvals = sel_sigmu[idx]['z']
                zmax = 1.1
                zstep = 0.03
                bins = np.arange(zmin, zmax+zstep, zstep)
                io = bins <= zmax
                bins = bins[io]
                """
                idxx = sel_sigmu['z'] >= zcomp
                sel_sigmub = sel_sigmu[idxx]
                zmax = 1.1
                zst = np.mean(np.diff(sel_sigmub['z']))/2
                zmin = sel_sigmub['z'].min()
                zmax += 2*zst
                bins = np.arange(zmin, zmax, 2.*zst)

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

        return df_tot

    def add_bias_x1_color(self, df_tot):
        df_fi = pd.DataFrame()
        for dbName in df_tot['dbName'].unique():
            idx = df_tot['dbName'] == dbName
            sel = df_tot[idx].copy()
            idm = self.sigma_bias_x1_color['dbName'] == dbName
            sel_bias = self.sigma_bias_x1_color[idm]
            if not sel_bias.empty:
                interpo = interp1d(
                    sel_bias['z'], sel_bias['delta_mu_bias'], bounds_error=False, fill_value=0.)
                zrange = sel['z_SN'].to_list()
                sel.loc[:, 'sigma_bias_x1_color'] = interpo(zrange)
            df_fi = pd.concat((df_fi, sel))

        return df_fi

    def add_sigbias_resimulate(self, data_sn, sigmaInt):

        # add a column sigma_bias_stat for all
        data_sn['sigma_bias_stat'] = 0.0
        data_sn['sigma_bias_x1_color'] = 0.0
        data_sn['sigma_mu_photoz'] = 0.0
        # add bias stat
        data_sn = self.add_bias_stat(data_sn)
        # add bias x1_color
        data_sn = self.add_bias_x1_color(data_sn)
        # add sigma_mu_photoz
        data_sn = self.add_sigma_photoz(data_sn, check_plot=False)

        # plot to cross-check
        """
        import matplotlib.pyplot as plt
        plt.plot(data_sn['z_SN'], data_sn['sigma_mu_photoz'], 'ko')
        plt.show()
        """
        # re-estimate the distance moduli including the bias error - for DD only - faster

        data_sn['mu_SN'] = self. simu_mu_SN(data_sn, sigmaInt)
        return data_sn

    def simu_mu_SN(self, data, sigmaInt, H0=70.0, Om=0.3, w0=-1.0, wa=0.0):

        from sn_fom.cosmo_fit import CosmoDist
        from random import gauss
        cosmo = CosmoDist(H0=H0)
        dist_mu = cosmo.mu_astro(data['z_SN'], Om, w0, wa)
        sigmu = np.array(data['sigma_mu_SN'])
        sigbias = np.array(data['sigma_bias_stat'])
        #data['sigma_bias_x1_color'] *= 2
        sigx1color = np.array(data['sigma_bias_x1_color'])
        """
        import matplotlib.pyplot as plt
        plt.plot(data['z_SN'], sigx1color)
        plt.show()
        """
        sigphotz = np.array(data['sigma_mu_photoz'])
        mu = [gauss(dist_mu[i], np.sqrt(sigmu[i]**2+sigmaInt**2+sigbias[i]**2+sigx1color[i]**2+sigphotz[i]**2))
              for i in range(len(dist_mu))]

        return mu

    def add_sigma_photoz(self, data, check_plot=False):

        sigma_photoz = 0.
        data['sigma_photoz'] = sigma_photoz
        fieldNames = ['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS']
        nhost_season = dict(zip(fieldNames, [100, 100, 300, 300, 300]))
        z_photoz = [0.8, 0.8, 0.6, 0.6, 0.6]
        fzdict = dict(zip(fieldNames, z_photoz))
        # get photoz correction here
        if not self.sigma_mu_photoz.empty:
            # sigmu_interp = interp1d(
            #    self.sigma_mu_photoz['z'], self.sigma_mu_photoz['sigma_mu_photoz'], bounds_error=False, fill_value=0.)
            sigma_photoz = self.sigma_mu_photoz['sigma_photoz'].unique()[0]
            data['sigma_mu_photoz'] = self.sigmu_interp(data['z_SN'])
            data['sigma_photoz'] = sigma_photoz
            data = self.apply_sigma_photoz_new(data)
            # data = self.host_measurements(resa, n_host_obs)

        """
            data_n = pd.DataFrame()
            for fName, zv in fzdict.items():
                dd = self.apply_sigma_photoz(data, fName, zv, sigmu_interp)
                data_n = pd.concat((dd, data_n))
            data = pd.DataFrame(data_n)

        data['sigma_photoz'] = sigma_photoz
        """
        if check_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for fieldName in fieldNames:
                idx = data['fieldName'] == fieldName
                sel = data[idx]
                if not sel.empty:
                    sel = sel.sort_values(by=['z_SN'])
                    ax.plot(sel['z_SN'], sel['sigma_mu_photoz'],
                            label=fieldName, marker='o')
            ax.legend()
            plt.show()

        """
        if not self.sigma_mu_photoz.empty:
            print('aoooo', data.columns)
            idx = data['sigma_mu_photoz'] < 1.e-5
            idxb = data['sigma_mu_photoz'] < 1.e-5
            print('nsn with host-z measurement', len(data[idx]), len(data))
            idx &= data['dd_type'] == 'ultra_dd'
            print('nsn with host-z measurement - ultra', len(data[idx]))
            idxb &= data['dd_type'] == 'deep_dd'
            print('nsn with host-z measurement - deep', len(data[idxb]))
        """
        return data

    def info_photoz(self, nsn_WFD_yearly):
        # get the number of seasons for ultra and deep fields
        self.n_host_obs = {}
        self.n_host_obs['ultra_dd'] = self.get_nseasons(
            ['COSMOS', 'XMM-LSS'])*200
        self.n_host_obs['deep_dd'] = self.get_nseasons(
            ['CDFS', 'ELAIS', 'ADFS'])*500

        if nsn_WFD_yearly < 0:
            # total Subaru expected spectra for DD
            self.n_host_obs['ultra_dd'] = 2000
            # total 4MOST expected spectra for DD
            self.n_host_obs['deep_dd'] = 1000

        # Subaru host efficiency vs z
        rz_Sub = [0.0, 0.2, 1., 1.37, 2.16]
        reff_Sub = [1., 0.98, 0.79, 0.71, 0.0]
        host_Sub = self.host_efficiency(rz=rz_Sub, reff=reff_Sub)

        # 4MOST host efficiency vs z
        rz_4M = [0.0, 0.34, 0.6, 0.9, 1, 1.1]
        reff_4M = [0.83, 0.79, 0.39, 0.06, 0.03, 0.0]
        host_4M = self.host_efficiency(rz=rz_4M, reff=reff_4M)

        interp_Sub = interp1d(
            host_Sub['z'], host_Sub['host_effi'], bounds_error=False, fill_value=0.)
        interp_4M = interp1d(
            host_4M['z'], host_4M['host_effi'], bounds_error=False, fill_value=0.)

        self.interp_field['ultra_dd'] = interp_Sub
        self.interp_field['deep_dd'] = interp_4M

        self.sigmu_interp = interp1d(
            self.sigma_mu_photoz['z'], self.sigma_mu_photoz['sigma_mu_photoz'], bounds_error=False, fill_value=0.)

    def apply_sigma_photoz_new(self, data):

        bins = np.arange(0., 1.2, 0.05)
        binsb = np.arange(0.0, 1.2, 0.05)

        bins_center = 0.5*(binsb[1:]-binsb[:-1])
        norm = {}
        for dd_type in ['ultra_dd', 'deep_dd']:
            intt = self.interp_field[dd_type](bins_center)
            norm[dd_type] = self.n_host_obs[dd_type]/np.sum(intt)

        data['dd_type'] = 'deep_dd'
        idx = data['fieldName'] == 'COSMOS'
        if len(data[idx]) > 0:
            data.loc[idx, 'dd_type'] = 'ultra_dd'
        idx = data['fieldName'] == 'XMM-LSS'
        if len(data[idx]) > 0:
            data.loc[idx, 'dd_type'] = 'ultra_dd'

        nn_data = pd.DataFrame()
        for dd_type in data['dd_type'].unique():
            idx = data['dd_type'] == dd_type
            sel = data[idx]
            new_data = sel.groupby(pd.cut(sel.z_SN, bins)).apply(
                lambda x: self.apply_photoz(x, self.interp_field[dd_type], norm[dd_type]))
            newd = pd.DataFrame(new_data['z_SN'].to_list(), columns=['z_SN'])
            for col in new_data.columns:
                if col != 'z_SN' and col != 'index':
                    newd[col] = new_data[col].to_list()
            nn_data = pd.concat((nn_data, newd))
        """
        import matplotlib.pyplot as plt
        plt.plot(host_Sub['z'], host_Sub['host_effi'])
        plt.plot(host_4M['z'], host_4M['host_effi'])
        plt.show()
        """
        return nn_data

    def host_efficiency(self, rz, reff):

        df = pd.DataFrame(rz, columns=['z'])
        df['host_effi'] = reff

        return df

    def host_measurements(self, data, nums):

        res_df = pd.DataFrame()
        for dd_type in data['dd_type'].unique():
            idx = data['dd_type'] == dd_type
            sel = data[idx]
            sel = sel.reset_index()
            nn_host = int(nums[dd_type])
            part_host = sel.sample(n=nn_host)
            part_no_host = sel.drop(part_host.index)
            part_host['sigma_mu_photoz'] = 0.0
            new_df = pd.concat((part_host, part_no_host))
            res_df = pd.concat((res_df, new_df))

        print('ici cols', res_df.columns)
        return res_df

    def get_nseasons(self, fieldList):

        n_seasons = [0]
        for fieldName in fieldList:
            ii = self.config.fieldName == fieldName
            sel = self.config[ii]
            if len(sel) > 0:
                n_seasons.append(sel['nseasons'].to_list()[0])

        return np.max(n_seasons)

    def apply_photoz(self, grp, interp, norm):

        zmin = grp.name.left
        zmax = grp.name.right
        zmean = np.mean([zmin, zmax])
        # apply SN host efficiency curves
        # effi = interp(np.mean([zmin, zmax]))
        # nSN = len(grp)
        # n_host = int(effi*nSN)
        # print('from host efficiency', n_host, nSN)
        n_host = int(interp(zmean)*norm)

        if len(grp) == 0:
            return grp
        if len(grp) <= n_host:
            grp['sigma_mu_photoz'] = 0.
            return grp

        grp = grp.reset_index()
        # part_host = grp.sample(frac=effi)
        part_host = grp.sample(n=n_host)
        part_host['sigma_mu_photoz'] = 0.0
        part_no_host = grp.drop(part_host.index)

        new_grp = pd.concat((part_host, part_no_host))

        return new_grp

    def apply_sigma_photoz(self, data, fieldName, zsel, sigmu_interp):

        idx = data['fieldName'] == fieldName
        sel = pd.DataFrame(data[idx])
        if not sel.empty:
            z_SN = sel['z_SN']
            sel['sigma_mu_photoz'] = sigmu_interp(z_SN)
            idx = sel['z_SN'] <= zsel
            sel.loc[idx, 'sigma_mu_photoz'] = 0.0
        return sel


def fit_SN_deprecated(dbNames, config, fields, snType, sigmu, nsn_bias, sn_wfd=pd.DataFrame(), params_fit=['Om', 'w0', 'wa'], saveSN='', sigma_bias=0.01):

    # simulate supernovae here - DDF
    data_sn = simul_SN(dbNames, config, fields, snType,
                       sigmu, nsn_bias, saveSN=saveSN, sigma_bias=sigma_bias)

    # add WFD SN if required
    print('NSN DD:', len(data_sn), 'WFD:', len(sn_wfd))
    data_sn['snType'] = 'DD'
    if not sn_wfd.empty:
        sn_wfd['snType'] = 'WFD'
        sn_wfd['fieldName'] = 'WFD'
        data_sn = pd.concat((data_sn, sn_wfd))

    if saveSN != '':
        data_sn.to_hdf(saveSN, key='sn')

    # cosmo fit here
    SNID = ''
    if saveSN != '':
        SNID = saveSN.split('.hdf5')[0]

    # second step: fit the data

    print('to fit', data_sn)
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
    sigma_bias_x1_color = params['sigma_bias_x1_color']
    params_fit = pd.DataFrame()
    np.random.seed(123456+j)
    for i in index:
        if saveSN != '':
            saveSN_f = '{}/SN_{}.hdf5'.format(saveSN, i)
        fitpar = fit_SN(dbNames, config,
                        fields, snType,
                        sigma_mu_from_simu, nsn_bias,
                        sn_wfd, params_for_fit,
                        saveSN=saveSN_f,
                        sigma_bias_x1_color=sigma_bias_x1_color)
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
    sigma_bias_x1_color = params['sigma_bias_x1_color']
    sigmaInt = params['sigmaInt']
    binned_cosmology = params['binned_cosmology']
    surveyType = params['surveyType']
    sigma_mu_photoz = params['sigma_mu_photoz']
    nsn_WFD_yearly = params['nsn_WFD_yearly']
    nsn_WFD_hostz = params['nsn_WFD_hostz']
    nsn_WFD_hostz_yearly = params['nsn_WFD_hostz_yearly']
    year_survey = params['year_survey']

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
                           sigma_bias_x1_color=sigma_bias_x1_color,
                           binned_cosmology=binned_cosmology,
                           surveyType=surveyType,
                           sigma_mu_photoz=sigma_mu_photoz,
                           nsn_WFD_yearly=nsn_WFD_yearly,
                           nsn_WFD_hostz=nsn_WFD_hostz,
                           nsn_WFD_hostz_yearly=nsn_WFD_hostz_yearly,
                           year_survey=year_survey)

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

    zmin = 0.005
    zmax = sigmu_from_simu['z'].max()+2.*zst
    bins = np.arange(zmin, zmax+2.*zst, 2.*zst)

    # get SN from simu
    data_sn = select(loadSN(fileDir, dbName, 'WFD', nfich=nfich))

    idx = data_sn['z'] < 0.2
    data_sn = data_sn[idx]
    # get the effective (bias effect) number of expected SN per bin
    group = data_sn.groupby(pd.cut(data_sn.z, bins))
    nsn_simu = len(data_sn)
    plot_centers = (bins[: -1] + bins[1:])/2
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

        df = pd.DataFrame()
        for io, dbName in enumerate(self.dbNames):
            zmin = 0.035
            zmax = 1.235
            nbins = 25  # 25 for 0.05 z-bin
            nbins = 41
            if 'WFD' in dbName:
                zmin = 0.0
                zmax = 0.20
                nbins = 20

            snType = self.snTypes[io]
            data_sn = transformSN(self.fileDir, dbName,
                                  snType, self.alpha, self.beta, self.Mb)
            for vv in ['x1', 'color', 'mb']:
                vvb = 'sigma_{}'.format(vv)
                if vvb not in data_sn.columns:
                    vvc = 'Cov_{}{}'.format(vv, vv)
                    data_sn[vvb] = np.sqrt(data_sn[vvc])
            bdatat = pd.DataFrame()
            for var in ['mu', 'sigma_mu', 'sigma_color', 'sigma_x1', 'sigma_mb']:
                bdata = binned_data(zmin, zmax, nbins, data_sn, var)
                bdata = bdata.round({'z': 2})
                bdata = self.check_fill(bdata, '{}_mean'.format(var))
                if bdatat.empty:
                    bdatat = bdata
                else:
                    bdatat = bdatat.merge(bdata, left_on=['z'], right_on=['z'])

            """
            bdatab = binned_data(zmin, zmax, nbins, data_sn, 'mu')

            bdata = bdata.round({'z': 2})
            bdatab = bdatab.round({'z': 2})

            bdata = self.check_fill(bdata, 'sigma_mu_mean')
            bdatab = self.check_fill(bdatab, 'mu_mean')
            bdata = bdata.merge(bdatab, left_on=['z'], right_on=['z'])
            """
            bdatat['dbName'] = dbName
            df = pd.concat((df, bdatat))
        return df

    def check_fill(self, data, var):
        """
        Method to remove zeros (if any)

        Parameters
        ---------------
        data: array
           data to process

        """

        zvals = data['z'].to_list()

        data = data.fillna(-1)
        idx = data[var] > 0.
        sel = data[idx]

        interp = interp1d(sel['z'], sel[var],
                          bounds_error=False, fill_value=0.)

        data[var] = interp(zvals)

        return data

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

        print('loading', outName)
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
        data_sn_df = pd.DataFrame()

        for i, dbName in enumerate(self.dbNames):
            print('processing', dbName)
            # print('loading SN')
            data_sn = loadSN(self.fileDir, dbName, 'allSN')
            for field in self.fields:
                idx = self.config['fieldName'].isin([field])
                dd = self.getNSN_bias(
                    dbName, data_sn, self.config[idx], [field])
                data_sn_df = pd.concat((data_sn_df, dd))
                print('SN inter', dbName, len(dd))

        return data_sn_df

    def getNSN_bias(self, dbName, data_sn, config, fields):
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

        # get the effective (bias effect) number of expected SN per bin

        fieldName = fields[0]
        nsn_eff = nSN_bin_eff(data_sn, nsn_per_bin, fieldName)

        nsn_eff['fieldName'] = fieldName
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
            fig, ax = plt.subplots(figsize=(12, 9))
            zcomps = sel['zcomp'].unique()
            zcomps = ['0.90', '0.80', '0.70', '0.65']
            ls = dict(
                zip(zcomps, ['solid', 'dotted', 'dashed', 'dashdot']))
            fig.suptitle('{}'.format(field))
            print(type(sel))
            sel = sel.fillna(0)
            for zcomp in zcomps:
                idxb = sel['zcomp'] == zcomp
                selb = sel[idxb].to_records(index=False)
                # ax.plot(selb['z'], selb['nsn_eff'],
                #        label='$z_{complee}$'+'= {}'.format(zcomp))
                xnew = np.linspace(
                    np.min(selb['z']), np.max(selb['z']), 100)
                spl = make_interp_spline(
                    selb['z'], selb['nsn_eff'], k=3)  # type: BSpline
                print('NSN', zcomp, np.cumsum(selb['nsn_eff'])[-1])
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
