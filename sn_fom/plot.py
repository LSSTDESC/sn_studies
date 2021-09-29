import matplotlib.pyplot as plt
from . import np
from sn_fom.cosmo_fit import CosmoDist
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.ticker import NullFormatter


class plotStat:

    def __init__(self, params):

        self.params = params

    def plotFoM(self):

        io = -1
        r = []
        for index, row in self.params.iterrows():
            io += 1
            fom, rho = self.getFoM(row)
            r.append((io, fom, rho))

        res = np.rec.fromrecords(r, names=['iter', 'FoM', 'correl'])
        print(res)

        fig, ax = plt.subplots()
        ax.plot(res['iter'], res['FoM'])
        figb, axb = plt.subplots()
        idx = res['FoM'] <= 1000.
        sel = res[idx]
        axb.hist(sel['FoM'], bins=20, histtype='step')
        plt.show()

    def getFoM(self, params_fit):

        # get FoM
        sigma_w0 = np.sqrt(params_fit['Cov_w0_w0'])
        sigma_wa = np.sqrt(params_fit['Cov_wa_wa'])
        sigma_w0_wa = params_fit['Cov_w0_wa']

        fom, rho = FoM(sigma_w0, sigma_wa, sigma_w0_wa)

        return fom, rho


def FoM(sigma_w0, sigma_wa, sigma_w0_wa, coeff_CL=6.17):
    """
    Function to estimate the Figure of Merit (FoM)
    It is inversely proportional to the area of the error ellipse in the w0-wa plane

    Parameters
    ---------------
    sigma_w0: float
      w0 error
    sigma_wa: float
      wa error
    sigma_w0_wa: float
      covariance (w0,wa)
    coeff_CL: float, opt
      confidence level parameter for the ellipse area (default: 6.17=>95% C.L.)

    Returns
    ----------
    FoM: the figure of Merit
    rho: correlation parameter (w0,wa)


    """
    print('in FoM')
    rho = sigma_w0_wa/(sigma_w0*sigma_wa)
    print('rhrhrhrh', rho)
    # get ellipse parameters
    a, b = ellipse_axis(sigma_w0, sigma_wa, sigma_w0_wa)
    area = coeff_CL*a*b

    print('alors FoM', rho, a, b, area)
    return 1./area, rho


def ellipse_axis(sigx, sigy, sigxy):
    """
    Function to estimate ellipse axis

    Parameters
    ---------------
    sigx: float
      sigma_x
    sig_y: float
      sigma_y
    sigxy: float
      sigma_xy correlation

    Returns
    ----------
    (a,b) The two ellipse axis

    """

    comm_a = 0.5*(sigx**2+sigy**2)
    comm_b = 0.25*(sigx**2-sigy**2)**2-sigxy**2
    if comm_b < 0.:
        comm_b = 0.
    a_sq = comm_a+np.sqrt(comm_b)
    b_sq = comm_a-np.sqrt(comm_b)

    print('ellipse', sigx, sigy, sigxy, comm_a, comm_b)

    return np.sqrt(a_sq), np.sqrt(b_sq)


class plotHubbleResiduals(CosmoDist):
    """
    Class to plot Hubble residuals
    This class inherits from the CosmoDist class

    Parameters
    ---------------
    fDir: str
      location dir of the files to process
    dbName: str
       db Name to process
    zlim: float
      redshift completeness
    H0 : float,opt
      Hubble cte (default: 72.  # km.s-1.Mpc-1)
    c: float, opt
     speed of the light (default: = 299792.458  # km.s-1)

    """

    def __init__(self, fitparams, fichName, H0=72, c=299792.458, rescale_factor=1, var_FoM=['Om', 'w0']):
        super().__init__(H0, c)

        # load SN
        data = pd.read_hdf(fichName)

        self.data = data
        self.Z = data['z_fit']
        self.Mb = -2.5*np.log10(data['x0_fit'])+10.635
        Cov_mbmb = (2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']
        self.Mber = np.sqrt(Cov_mbmb)
        self.gx1 = data['Cov_x1x1']
        self.gxc = data['Cov_colorcolor']
        self.cov1 = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        # self.cov1 = data['Cov_x1mb']
        self.cov2 = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        # self.cov2 = data['Cov_colormb']
        self.cov3 = data['Cov_x1color']
        self.sigZ = data['z_fit']/(1.e5*data['z_fit'])
        self.X1 = data['x1_fit']
        self.X2 = data['color_fit']
        print('Number of SN', len(data))
        self.zmin = np.min(self.Z)
        self.zmax = np.max(self.Z)

        print(fitparams)
        self.Om = fitparams['Om']
        self.w0 = fitparams['w0']
        self.wa = fitparams['wa']
        # self.wa = 0.0
        self.alpha = fitparams['alpha']
        self.beta = fitparams['beta']
        self.M = fitparams['M']
        mu = -self.M+self.alpha*self.X1-self.beta*self.X2+self.Mb
        sigma_mu = np.sqrt(self.sigmI(self.alpha, self.beta))
        self.data['mu'] = mu
        self.data['sigma_mu'] = sigma_mu

        """
        self.sigma_w0 = np.sqrt(fitparams['Cov_w0_w0'])
        self.sigma_wa = np.sqrt(fitparams['Cov_wa_wa'])
        self.sigma_w0_wa = fitparams['Cov_w0_wa']
        """
        vara = var_FoM[0]
        sig_vara = 'Cov_{}_{}'.format(vara, vara)
        varb = var_FoM[1]
        varc = 'wa'
        sig_varb = 'Cov_{}_{}'.format(varb, varb)
        sig_varc = 'Cov_{}_{}'.format(varc, varc)
        sig_vara_varb = 'Cov_{}_{}'.format(vara, varb)
        sig_varb_varc = 'Cov_{}_{}'.format(varb, varc)

        self.sigma_a = np.sqrt(fitparams[sig_vara])
        self.sigma_b = np.sqrt(fitparams[sig_varb])
        self.sigma_c = np.sqrt(fitparams[sig_varc])
        self.sigma_a_b = fitparams[sig_vara_varb]
        self.sigma_b_c = fitparams[sig_varb_varc]

        self.chi2 = fitparams['chi2']/fitparams['ndf']

        self.lega = '$\Omega_m$'+' = {}'.format(np.round(fitparams[vara], 3))
        self.lega += '$\pm $'+'{}'.format(np.round(self.sigma_a, 3))
        self.legb = '$w_0$'+' = {}'.format(np.round(fitparams[varb], 3))
        self.legb += '$\pm $'+'{}'.format(np.round(self.sigma_b, 3))
        self.legc = '$w_a$'+' = {}'.format(np.round(fitparams[varc], 3))
        self.legc += '$\pm $'+'{}'.format(np.round(self.sigma_c, 3))

    def plots(self):

        # FoM_val, rho = FoM(self.sigma_a, self.sigma_b, self.sigma_a_b)
        FoM_val, rho = FoM(self.sigma_b, self.sigma_c, self.sigma_b_c)
        fig = plt.figure(figsize=(12, 8))
        ttit = 'FoM (95%)  = {} \n'.format(np.round(FoM_val, 2))
        ttit += '{} {} {} \n'.format(self.lega, self.legb, self.legc)
        """
        ttit += '$\sigma_{w_0}$'+'= {}'.format(np.round(self.sigma_w0,3))
        ttit += '$\sigma_{w_a}$'+'= {}'.format(np.round(self.sigma_wa,3))
        """
        ttit += ' $\chi^2/ndf$'+' = {}'.format(np.round(self.chi2, 5))
        fig.suptitle(ttit)

        ax = fig.add_axes((.1, .3, .8, .6))
        self.plot_hubble(ax)
        print('zmin and zmax', self.zmin, self.zmax)
        ax.set_xlim([self.zmin, self.zmax])
        axb = fig.add_axes((.1, .1, .8, .2))
        bottom_h = left_h = 0.1 + 0.8 + 0.02
        rect_histy = [left_h, 0.1, 0.05, 0.2]
        axbh = fig.add_axes(rect_histy)
        nullfmt = NullFormatter()
        axbh.yaxis.set_major_formatter(nullfmt)
        self.plot_residuals(axb, axbh, binned=True)
        axb.set_xlim([self.zmin, self.zmax])

        # plt.show()

    def sigmI(self, alpha, beta):

        # return self.Mber**2+(Xi1**2)*self.gx1+(Xi2**2)*self.gxc-2*Xi1*self.cov1-2*Xi2*self.cov2+2*Xi1*Xi2*self.cov3
        return self.Mber**2+(alpha**2)*self.gx1+(beta**2)*self.gxc+2*alpha*self.cov1-2*beta*self.cov2-2*alpha*beta*self.cov3

    def plot_hubble(self, ax):
        ax.errorbar(self.Z, self.Mb+self.alpha*self.X1-self.beta*self.X2,
                    yerr=np.sqrt(self.sigmI(self.alpha, self.beta)), xerr=None, fmt='.', color='k')
        z = np.arange(0.001, 1., 0.001)
        r = self.mu(z, self.Om, self.w0, self.wa)
        ax.plot(z, r+self.M, color='r')

    def plot_sn_vars(self):
        """
        plot data related to supernovae: x1, color and diff with fits

        """

        fig, ax = plt.subplots(ncols=3, nrows=2)

        vars = ['x1', 'x1_fit', 'color', 'color_fit']

        ax[0, 0].hist(self.data['x1'], bins=100, histtype='step')
        ax[0, 1].hist(self.data['color'], bins=100, histtype='step')
        ax[0, 2].hist(self.data['z'], bins=100, histtype='step')

        idx = self.data['fitstatus'] == 'fitok'
        # idx &= np.sqrt(self.data_all['Cov_colorcolor']) <= 0.04
        sel = self.data[idx]
        ax[1, 0].hist(sel['x1']-sel['x1_fit'],
                      bins=20, histtype='step')
        ax[1, 1].hist(sel['color'] -
                      sel['color_fit'], bins=20, histtype='step')
        # ax[1, 2].hist(sel['z'], bins=20, histtype='step')
        ax[1, 2].plot(sel['z'], np.sqrt(sel['Cov_colorcolor']), 'ko')

    def plot_residuals(self, axb, axbh, binned=False, nbins=50):
        """
        Method to plot distance modulus vs redshift

        Parameters
        ---------------
        data: pandas df
         data to plot

        """

        r = []

        for z in np.arange(self.zmin, self.zmax+0.01, 0.001):
            mu, mup, mum = self.mu(z, self.Om, self.w0, self.wa), self.mu(
                z, self.Om, self.w0+0.01, self.wa), self.mu(z, self.Om, self.w0-0.01, self.wa)
            r.append((z, mu, mup, mum))

        res = np.rec.fromrecords(r, names=['z', 'mu', 'mup', 'mum'])

        """
        # fix, ax = plt.subplots()
        fig = plt.figure()
        # figb, axb = plt.subplots()
        ax = fig.add_axes((.1, .3, .8, .6))

        ax.plot(res['z'], res['mu'], color='r')
        ax.plot(res['z'], res['mup'], color='b')
        ax.plot(res['z'], res['mum'], color='b')
        """

        res_interp = interp1d(res['z'], res['mu'],
                              bounds_error=False, fill_value=0.)

        # add the mu columns
        mu = -self.M+self.alpha*self.X1-self.beta*self.X2+self.Mb
        sigma_mu = np.sqrt(self.sigmI(self.alpha, self.beta))
        # add the mu_th column
        mu_th = res_interp(self.Z)

        # residuals: mu-mu_th/mu
        mu_residual = mu_th-mu

        x, y, yerr, residuals = 0., 0., 0., 0.
        if binned:
            x, y, yerr, residuals = self.binned_data(
                self.zmin, self.zmax, self.data, nbins, res_interp)
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        else:
            x, y, yerr = self.Z, mu, sigma_mu
            residuals = mu_residual
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        """
        ax.errorbar(x, y, yerr=yerr,
                    color='k', lineStyle='None', marker='o', ms=2)
        ax.grid()
        """
        # axb = fig.add_axes((.1, .1, .8, .2))
        axb.errorbar(x, residuals, yerr=None, color='k',
                     lineStyle='None', marker='o', ms=2)
        axbh.hist(residuals, bins=20, orientation='horizontal')
        print('Residuals', np.mean(residuals), np.std(residuals))
        axb.errorbar(res['z'], res['mu']-res['mup'], color='r', ls='dotted')
        axb.errorbar(res['z'], res['mu']-res['mum'], color='r', ls='dotted')

        axb.grid()

    def binned_data(self, zmin, zmax, data, nbins, muth_interp, vary='mu', erry='sigma_mu'):
        """
        Method to transform a set of data to binned data

        Parameters
        ---------------
        zmin: float
          min redshift
        zmax: float
          max redshift
        data: pandas df
          data to be binned
        vary: str, opt
          y-axis variable (default: mu)
        erry: str, opt
          y-axis var error (default: sigma_mu)

        Returns
        -----------
        x, y, yerr:
        x : redshift centers
        y: weighted mean of distance modulus
        yerr: distance modulus error

        """
        data['diff_mu'] = muth_interp(data['z'])-data[vary]
        bins = np.linspace(zmin, zmax, nbins)
        group = data.groupby(pd.cut(data.z, bins))
        plot_centers = (bins[:-1] + bins[1:])/2
        plot_values = group.mu.mean()
        residuals = group.diff_mu.mean()
        # plot_values = group.apply(lambda x: np.sum(
        #    x[vary]/x[erry]**2)/np.sum(1./x[erry]**2))
        print(plot_values)
        error_values = group.apply(
            lambda x: 1./np.sqrt(np.sum(1./x[erry]**2)))
        print('error', error_values)

        return plot_centers, plot_values, error_values, residuals


class plotHubbleResiduals_mu(CosmoDist):
    """
    Class to plot Hubble residuals
    This class inherits from the CosmoDist class

    Parameters
    ---------------
    fDir: str
      location dir of the files to process
    dbName: str
       db Name to process
    zlim: float
      redshift completeness
    H0 : float,opt
      Hubble cte (default: 72.  # km.s-1.Mpc-1)
    c: float, opt
     speed of the light (default: = 299792.458  # km.s-1)

    """

    def __init__(self, fitparams, fichName, H0=70., c=299792.458, rescale_factor=1, var_FoM=['Om', 'w0'], var_fit=['Om', 'w0', 'wa']):
        super().__init__(H0, c)

        # load SN
        data = pd.read_hdf(fichName)

        self.data = data
        self.Z = data['z_SN']
        self.mu = data['mu_SN']
        self.sigma_mu = data['sigma_mu_SN']
        self.zmin, self.zmax = np.min(self.Z), np.max(self.Z)

        print(fitparams)
        self.Om = fitparams['Om']
        self.w0 = fitparams['w0']
        self.wa = fitparams['wa']

        sig = {}
        for vv in var_fit:
            sig[vv] = {}
            sig[vv]['val'] = fitparams[vv]
            sig[vv]['sigma'] = np.sqrt(fitparams['Cov_{}_{}'.format(vv, vv)])

        vara = var_FoM[0]
        sig_vara = 'Cov_{}_{}'.format(vara, vara)
        varb = var_FoM[1]

        sig_varb = 'Cov_{}_{}'.format(varb, varb)
        sig_vara_varb = 'Cov_{}_{}'.format(vara, varb)

        self.sigma_a = np.sqrt(fitparams[sig_vara])
        self.sigma_b = np.sqrt(fitparams[sig_varb])
        self.sigma_a_b = fitparams[sig_vara_varb]

        self.chi2 = fitparams['chi2']/fitparams['ndf']

        """
        self.lega = '$\Omega_m$'+' = {}'.format(np.round(fitparams[vara], 3))
        self.lega += '$\pm $'+'{}'.format(np.round(self.sigma_a, 3))
        self.legb = '$w_0$'+' = {}'.format(np.round(fitparams[varb], 3))
        self.legb += '$\pm $'+'{}'.format(np.round(self.sigma_b, 3))
        """
        self.lega = '$\Omega_m$'+' = {}'.format(np.round(sig['Om']['val'], 3))
        self.lega += '$\pm $'+'{}'.format(np.round(sig['Om']['sigma'], 3))
        self.lega += ' $w_0$'+' = {}'.format(np.round(sig['w0']['val'], 3))
        self.lega += '$\pm $'+'{}'.format(np.round(sig['w0']['sigma'], 3))
        if 'wa' in var_fit:
            self.lega += ' $w_a$'+' = {}'.format(np.round(sig['wa']['val'], 3))
            self.lega += '$\pm $'+'{}'.format(np.round(sig['wa']['sigma'], 3))

    def plots(self):

        # FoM_val, rho = FoM(self.sigma_a, self.sigma_b, self.sigma_a_b)
        FoM_val, rho = FoM(self.sigma_a, self.sigma_b, self.sigma_a_b)
        fig = plt.figure(figsize=(12, 8))
        ttit = 'FoM (95%)  = {} \n'.format(np.round(FoM_val, 2))
        ttit += '{} \n'.format(self.lega)
        """
        ttit += '$\sigma_{w_0}$'+'= {}'.format(np.round(self.sigma_w0,3))
        ttit += '$\sigma_{w_a}$'+'= {}'.format(np.round(self.sigma_wa,3))
        """
        ttit += ' $\chi^2/ndf$'+' = {}'.format(np.round(self.chi2, 5))
        fig.suptitle(ttit)

        ax = fig.add_axes((.1, .3, .8, .6))
        self.plot_hubble(ax)
        print('zmin and zmax', self.zmin, self.zmax)
        ax.set_xlim([self.zmin, self.zmax])
        axb = fig.add_axes((.1, .1, .8, .2))
        bottom_h = left_h = 0.1 + 0.8 + 0.02
        rect_histy = [left_h, 0.1, 0.05, 0.2]
        axbh = fig.add_axes(rect_histy)
        nullfmt = NullFormatter()
        axbh.yaxis.set_major_formatter(nullfmt)
        self.plot_residuals(axb, axbh, binned=True)
        axb.set_xlim([self.zmin, self.zmax])

        # plt.show()
    def plot_hubble(self, ax):
        ax.errorbar(self.Z, self.mu,
                    yerr=self.sigma_mu, xerr=None, fmt='.', color='k')
        z = np.arange(0.001, 1., 0.001)
        r = self.mu_astro(z, self.Om, self.w0, self.wa)
        ax.plot(z, r, color='r')

    def plot_residuals(self, axb, axbh, binned=False, nbins=50):
        """
        Method to plot distance modulus vs redshift

        Parameters
        ---------------
        data: pandas df
         data to plot

        """

        r = []

        for z in np.arange(self.zmin, self.zmax+0.01, 0.001):
            mu, mup, mum = self.mu_astro(z, self.Om, self.w0, self.wa), self.mu_astro(
                z, self.Om, self.w0+0.01, self.wa), self.mu_astro(z, self.Om, self.w0-0.01, self.wa)
            r.append((z, mu, mup, mum))

        res = np.rec.fromrecords(r, names=['z', 'mu', 'mup', 'mum'])

        res_interp = interp1d(res['z'], res['mu'],
                              bounds_error=False, fill_value=0.)

        # add the mu_th column
        mu_th = res_interp(self.Z)

        # residuals: mu-mu_th/mu
        mu_residual = mu_th-mu

        x, y, yerr, residuals = 0., 0., 0., 0.
        if binned:
            x, y, yerr, residuals = self.binned_data(
                self.zmin, self.zmax, self.data, nbins, res_interp)
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        else:
            x, y, yerr = self.Z, self.mu, self.sigma_mu
            residuals = mu_residual
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        """
        ax.errorbar(x, y, yerr=yerr,
                    color='k', lineStyle='None', marker='o', ms=2)
        ax.grid()
        """
        # axb = fig.add_axes((.1, .1, .8, .2))
        axb.errorbar(x, residuals, yerr=None, color='k',
                     lineStyle='None', marker='o', ms=2)
        axbh.hist(residuals, bins=20, orientation='horizontal')
        print('Residuals', np.mean(residuals), np.std(residuals))
        axb.errorbar(res['z'], res['mu']-res['mup'], color='r', ls='dotted')
        axb.errorbar(res['z'], res['mu']-res['mum'], color='r', ls='dotted')

        axb.grid()

    def binned_data(self, zmin, zmax, data, nbins, muth_interp, vary='mu_SN', erry='sigma_mu_SN'):
        """
        Method to transform a set of data to binned data

        Parameters
        ---------------
        zmin: float
          min redshift
        zmax: float
          max redshift
        data: pandas df
          data to be binned
        vary: str, opt
          y-axis variable (default: mu)
        erry: str, opt
          y-axis var error (default: sigma_mu)

        Returns
        -----------
        x, y, yerr:
        x : redshift centers
        y: weighted mean of distance modulus
        yerr: distance modulus error

        """
        data['diff_mu'] = muth_interp(data['z_SN'])-data[vary]
        bins = np.linspace(zmin, zmax, nbins)
        group = data.groupby(pd.cut(data.z_SN, bins))
        plot_centers = (bins[:-1] + bins[1:])/2
        plot_values = group.mu_SN.mean()
        residuals = group.diff_mu.mean()
        # plot_values = group.apply(lambda x: np.sum(
        #    x[vary]/x[erry]**2)/np.sum(1./x[erry]**2))
        print(plot_values)
        error_values = group.apply(
            lambda x: 1./np.sqrt(np.sum(1./x[erry]**2)))
        print('error', error_values)

        return plot_centers, plot_values, error_values, residuals


def binned_data(zmin, zmax, data, nbins, vary='mu', erry='sigma_mu'):
    """
    Method to transform a set of data to binned data

    Parameters
    ---------------
    zmin: float
        min redshift
    zmax: float
        max redshift
    data: pandas df
        data to be binned
    vary: str, opt
        y-axis variable (default: mu)
    erry: str, opt
        y-axis var error (default: sigma_mu)

    Returns
    -----------
    x, y, yerr:
    x : redshift centers
    y: weighted mean of distance modulus
    yerr: distance modulus error

    """

    bins = np.linspace(zmin, zmax, nbins)
    if zmin < 1.e-8:
        bins[0] = 0.01
    group = data.groupby(pd.cut(data.z, bins))
    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group[vary].mean()

    # plot_values = group.apply(lambda x: np.sum(
    #    x[vary]/x[erry]**2)/np.sum(1./x[erry]**2))
    print('plot values', plot_values)
    error_values = None
    if erry != '':
        error_values = group.apply(
            lambda x: 1./np.sqrt(np.sum(1./x[erry]**2)))
        print('error', error_values)

    return plot_centers, plot_values, error_values


def plotFitRes(data):
    """
    Function to plot fit results

    Parameters
    ---------------
    data: pandas df
      data to plot
    """

    # get the FoM
    var_a = 'Om'
    var_b = 'w0'
    Cov_a = 'Cov_{}_{}'.format(var_a, var_a)
    Cov_b = 'Cov_{}_{}'.format(var_b, var_b)
    Cov_a_b = 'Cov_{}_{}'.format(var_a, var_b)
    sigma_a = 'sigma_{}'.format(var_a)
    sigma_b = 'sigma_{}'.format(var_b)
    data[sigma_a] = np.sqrt(data[Cov_a])
    data[sigma_b] = np.sqrt(data[Cov_b])

    print('hhhhhhhhhkokokoo', data[sigma_a], data[sigma_b])
    data['FoM'] = data.apply(lambda x: FoM(
        x[sigma_a], x[sigma_b], x[Cov_a_b])[0], axis=1)
    print('FoM', data['FoM'])
    fig, ax = plt.subplots(ncols=2, nrows=2)
    idx = data['FoM'] < 1.e10
    sel = data[idx]
    ax[0, 0].hist(sel[sigma_a], histtype='step', bins=100)
    ax[0, 1].hist(sel[sigma_b], histtype='step', bins=100)

    ax[1, 0].hist(sel['FoM'], histtype='step', bins=50)

    print('medians', data[[sigma_a, sigma_b, 'FoM']].median())

    plt.show()


class plotSN:

    def __init__(self, params_fit, params_true):

        self.params_fit = params_fit
        self.ppt = params_true

        from sn_fom.cosmo_fit import CosmoDist

        self.cosmo = CosmoDist()

    def __call__(self):

        # from sn_fom.cosmo_fit import FitData
        fig, ax = plt.subplots()
        ip = -1
        for i, row in self.params_fit.iterrows():
            ip += 1
            snName = '{}.hdf5'.format(row['SNID'])
            data = pd.read_hdf(snName)
            print(data[['z', 'z_fit']])
            data = self.complete_data(data)
            pp = []
            for vv in ['Om', 'w0', 'wa']:
                var = 'sigma_{}'.format(vv)
                row[var] = np.sqrt(row['Cov_{}_{}'.format(vv, vv)])
                pp.append(var)

            print(row[pp])
            # self.binned(data, ax, var='resi_mu', error='sigma_mu')
            self.binned(data, ax, var='x1', error='')
            """
            print('fitted', row)
            fit = FitData(data)

            newpar = fit()

            print('new fit', newpar)
            print(test)
            """

            # if ip >= 1:
            #    break

        plt.show()
        print(test)

    def complete_data(self, data):

        data['Mb'] = -2.5*np.log10(data['x0_fit'])+10.635
        data['Cov_mbmb'] = (
            2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']

        data['Cov_x1mb'] = -2.5*data['Cov_x0x1']/(data['x0_fit']*np.log(10))

        data['Cov_colormb'] = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))

        data['sigma_mu'] = data['Cov_mbmb']
        + self.ppt['alpha']**2*data['Cov_x1x1']
        + self.ppt['beta']**2*data['Cov_colorcolor']
        + 2 * self.ppt['alpha']*data['Cov_x1mb']
        - 2.*self.ppt['beta']*data['Cov_colormb']
        - 2.*self.ppt['alpha']*self.ppt['beta']*data['Cov_x1color']

        data['sigma_mu'] = np.sqrt(data['sigma_mu'])
        data['mu'] = -self.ppt['M']+data['mbfit']+self.ppt['alpha'] * \
            data['x1_fit']-self.ppt['beta']*data['color_fit']

        data['resi_mu'] = self.cosmo.mu(
            data['z'], self.ppt['Om'], self.ppt['w0'], self.ppt['wa'])-data['mu']
        return data

    def binned(self, data, ax, var='mu', error=''):

        zmin, zmax = 0.05, 1.
        nbins = 20
        bins = np.linspace(zmin, zmax, nbins)
        group = data.groupby(pd.cut(data.z, bins))
        plot_centers = (bins[:-1] + bins[1:])/2

        print(group.size(), np.sum(group.size()))
        if var == 'size':
            plot_values = group.size()
        else:
            plot_values = group[var].median()

        error_values = None
        if error != '':
            error_values = group.apply(
                lambda x: 1./np.sqrt(np.sum(1./x[error]**2)))

        ax.errorbar(plot_centers, plot_values, yerr=error_values, marker='o')
