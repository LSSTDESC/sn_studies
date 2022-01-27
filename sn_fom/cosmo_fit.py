from sn_fom.utils import loadSN, loadData, select, selSN
from . import np
from sn_tools.sn_utils import multiproc
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy import optimize
from sn_tools.sn_calcFast import faster_inverse
import pandas as pd
import time
import copy
import scipy.linalg as la
from iminuit import Minuit, describe, cost
from astropy.cosmology import w0waCDM


class zcomp_pixels:
    """
    class to estimate the redshift of completeness of fully simu+fit data

    Parameters
    ---------------
    dirFile: str
      location directory of the files
    dbName: str
      OS to process
    tagprod: str
      tag for the files

    """

    def __init__(self, dirFile, dbName, tagprod):

        self.data = loadData(dirFile, dbName, tagprod)
        print(len(self.data))
        print('Npixels', len(np.unique(self.data['healpixID'])))
        import healpy as hp
        print('pixel size', hp.nside2pixarea(128, degrees=True))

    def __call__(self):

        pixels = np.unique(self.data['healpixID'])
        params = {}

        res = multiproc(pixels, params, self.zcomp, 4)

        return res

    def zcomp(self, pixel_list, params={}, j=0, output_q=None):
        """
        Method to estimate the redshift limit for a set of pixels

        Parameters
        ---------------
        pixel_list: list(int)
          list of pixels to consider
        params: dict, opt
          parameters of the method (default: {})
        j: int, opt
          multiprocessing tag (default: 0)
        output_q: multiprocessing queue (opt)
          queue for multiprocessing
        """
        idx = self.data['healpixID'].isin(pixel_list)

        sel = self.data[idx]

        res = sel.groupby(['healpixID', 'season']).apply(
            lambda x: self.zcomp095(x)).reset_index()

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def zcomp095(self, grp):
        """
        Method to estimate the 95% redshift completeness from the cumulative

        Parameters
        ---------------
        grp: pandas df group
          data to process

        Returns
        -----------
        zcomp095: float
        """

        """
        idxb = grp['fitstatus'] == 'fitok'
        idxb &= np.sqrt(grp['Cov_colorcolor']) <= 0.04
        selb = grp[idxb].to_records(index=False)
        """
        print(np.unique(grp['fitstatus']))
        selb = select(grp).to_records(index=False)
        selb.sort(order=['z'])
        print('after sel', len(selb))
        if len(selb) >= 2:
            norm = np.cumsum(selb['z'])[-1]
            zlim = interp1d(
                np.cumsum(selb['z'])/norm, selb['z'], bounds_error=False, fill_value=0.)
            return pd.DataFrame({'zcomp': [zlim(0.95)]})
        return pd.DataFrame()


class CosmoDist:
    """
    class to estimate cosmology parameters

    Parameters
    ---------------
    H0 : float,opt
      Hubble cte (default: 72.  # km.s-1.Mpc-1)
    c: float, opt
     speed of the light (default: = 299792.458  # km.s-1)

    """

    def __init__(self, H0=70, c=2.99792e5):

        self.H0 = H0
        self.c = c

    def dL(self, z, Om=0.3, w0=-1., wa=0.0):

        cosmology = w0waCDM(H0=self.H0,
                            Om0=Om,
                            Ode0=1.-Om,
                            w0=w0, wa=wa)

        return cosmology.luminosity_distance(z).value*1.e6

    def cosmo_func(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate the integrand for the luminosity distance

        Parameters
        ---------------
        z: float
          redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
         w0 DE parameter (default: -1.0)
        wa: float, opt
          wa DE parameter (default: 0.)

        Returns
        -----------
        the integrand (float)

        """
        wp = w0+wa*z/(1.+z)
        # wp = w0

        H = Om*(1+z)**3+(1.-Om)*(1+z)**(3*(1.+wp))
        # H = Om*(1+z)**3+(1.-Om)*(1+z)

        fu = np.sqrt(H)

        return 1/fu

    def dL_old(self, z, Om=0.3, w0=-1., wa=0.0):
        """
        Method to estimate the luminosity distance

        Parameters
        ---------------
      z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
         w0 DE parameter (default: -1.0)
       wa: float, opt
         wa DE parameter (default: 0.)

        Returns
        ----------
        luminosity distance
        """
        norm = self.c/self.H0
        norm *= 1.e6

        def integrand(x): return self.integrand(x, norm, Om, w0, wa)

        if (hasattr(z, '__iter__')):
            s = np.zeros(len(z))
            for i, t in enumerate(z):
                s[i] = (1+t)*quad(integrand, 0.0, t, limit=100)[0]
            return s
        else:
            return (1+z)*quad(integrand, 0.0, z, limit=100)[0]

    def integrand(self, z, norm, Om, w0, wa):

        return norm*self.cosmo_func(z, Om, w0, wa)

    def mu_old(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate distance modulus

        Parameters
        ---------------
        z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
          w0 DE parameter (default: -1.0)
        wa: float, opt
            wa DE parameter (default: 0.)

        Returns
        -----------
        distance modulus (float)

        """

        if (hasattr(z, '__iter__')):
            return np.log10(self.dL(z, Om, w0, wa))*5-5
        else:
            return (np.log10(self.dL([z], Om, w0, wa))*5-5)[0]

        # return 5.*np.log10(self.dL(z, Om, w0, wa))+25. #if dL in Mpc

    def mu(self, z, Om=0.3, w0=-1.0, wa=0.0):
        """
        Method to estimate distance modulus

        Parameters
        ---------------
        z: float
           redshift
        Om: float, opt
          Omega_m parameter (default: 0.3)
        w0: float, opt
          w0 DE parameter (default: -1.0)
        wa: float, opt
            wa DE parameter (default: 0.)

        Returns
        -----------
        distance modulus (float)

        """

        if (hasattr(z, '__iter__')):
            return np.log10(self.dL(z, Om, w0, wa))*5-5
        else:
            return (np.log10(self.dL([z], Om, w0, wa))*5-5)[0]

    def mu_astro(self, z, Om, w0, wa):

        cosmology = w0waCDM(H0=self.H0,
                            Om0=Om,
                            Ode0=1.-Om,
                            w0=w0, wa=wa)

        return cosmology.distmod(z).value

    def mufit(self, z, alpha, beta, Mb, x1, color, mbfit, Om=0.3, w0=-1.0, wa=0.0):

        return mbfit+alpha*x1-beta*color-self.mu(z, Om, w0, wa)-Mb


class FoM(CosmoDist):
    """
    Class to estimate a statistical FoM
    This class inherits from the CosmoDist class

    Parameters
    ---------------
    fDir: str
      location dir of the files to process
    dbName: str
       db Name to process
    tagprod: str
       tag for the production
    zlim: pandas df
      redshift limit (per HEALPixel and season)
    H0 : float,opt
      Hubble cte (default: 72.  # km.s-1.Mpc-1)
    c: float, opt
     speed of the light (default: = 299792.458  # km.s-1)

    """

    def __init__(self, fDir, dbName, tagprod, zlim, H0=72, c=299792.458, rescale_factor=1):
        super().__init__(H0, c)

        # load data
        data = loadData(fDir, dbName, tagprod)

        # select data according to (zlim, season)

        data = data.merge(zlim, left_on=['healpixID', 'season'], right_on=[
            'healpixID', 'season'])

        self.data_all = data

        data = select(data)
        idx = data['z']-data['zcomp'] <= 0.
        idx = np.abs(data['z']-data['zcomp']) >= 0.

        self.data = data[idx].copy()
        # print('Number of SN', len(self.data))
        self.NSN = int(len(self.data)/rescale_factor)

    def plot_sn_vars(self):
        """
        plot data related to supernovae: x1, color and diff with fits

        """

        fig, ax = plt.subplots(ncols=3, nrows=2)

        vars = ['x1', 'x1_fit', 'color', 'color_fit']

        ax[0, 0].hist(self.data_all['x1'], bins=100, histtype='step')
        ax[0, 1].hist(self.data_all['color'], bins=100, histtype='step')
        ax[0, 2].hist(self.data_all['z'], bins=100, histtype='step')

        idx = self.data_all['fitstatus'] == 'fitok'
        # idx &= np.sqrt(self.data_all['Cov_colorcolor']) <= 0.04
        sel = self.data_all[idx]
        ax[1, 0].hist(sel['x1']-sel['x1_fit'],
                      bins=20, histtype='step')
        ax[1, 1].hist(sel['color'] -
                      sel['color_fit'], bins=20, histtype='step')
        # ax[1, 2].hist(sel['z'], bins=20, histtype='step')
        ax[1, 2].plot(sel['z'], np.sqrt(sel['Cov_colorcolor']), 'ko')

    def plot_data_cosmo(self, data, alpha=0.14, beta=3.1, Mb=-18.8, binned=False, nbins=50):
        """
        Method to plot distance modulus vs redshift

        Parameters
        ---------------
        data: pandas df
         data to plot

        """

        zmax = np.max(data['z'])
        zmin = np.min(data['z'])
        r = []

        for z in np.arange(zmin, zmax+0.01, 0.001):
            mu, mup, mum = self.mu(z), self.mu(
                z, w0=-1.0+0.01), self.mu(z, w0=-1.0-0.01)
            r.append((z, mu, mup, mum))

        res = np.rec.fromrecords(r, names=['z', 'mu', 'mup', 'mum'])

        # fix, ax = plt.subplots()
        fig = plt.figure()
        # figb, axb = plt.subplots()
        ax = fig.add_axes((.1, .3, .8, .6))

        ax.plot(res['z'], res['mu'], color='r')
        ax.plot(res['z'], res['mup'], color='b')
        ax.plot(res['z'], res['mum'], color='b')

        res_interp = interp1d(res['z'], res['mu'],
                              bounds_error=False, fill_value=0.)

        print(data.columns)
        # add the mu columns
        data['mu'] = data['mbfit']+alpha * \
            data['x1_fit']-beta*data['color_fit']-Mb

        # add the mu_th column
        data['mu_th'] = res_interp(data['z'])

        # residuals: mu-mu_th/mu
        data['mu_residual'] = (data['mu_th']-data['mu'])

        x, y, yerr, residuals = 0., 0., 0., 0.
        if binned:
            x, y, yerr, residuals = self.binned_data(
                zmin, zmax, data, nbins, res_interp)
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        else:
            pp = data.to_records(index=False)
            x, y, yerr = pp['z'], pp['mu'], pp['sigma_mu']
            residuals = pp['mu_residual']
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        ax.errorbar(x, y, yerr=yerr,
                    color='k', lineStyle='None', marker='o', ms=2)
        ax.grid()

        axb = fig.add_axes((.1, .1, .8, .2))
        axb.errorbar(x, residuals, yerr=None, color='k',
                     lineStyle='None', marker='o', ms=2)
        axb.errorbar(res['z'], res['mu']-res['mup'], color='r', ls='dotted')
        axb.errorbar(res['z'], res['mu']-res['mum'], color='r', ls='dotted')

        axb.grid()
        plt.show()

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


def deriv(grp, fom, params, epsilon):

    pplus = {}
    pminus = {}
    for key in params.keys():
        pplus[key] = params[key]+epsilon[key]
        pminus[key] = params[key]-epsilon[key]

    vva = fom.mufit(grp['z'], pplus['alpha'], pplus['beta'], pplus['Mb'], grp['x1_fit'],
                    grp['color_fit'], grp['mbfit'], pplus['Om'], pplus['w0'], pplus['wa'])

    vvb = fom.mufit(grp['z'], pminus['alpha'], pminus['beta'], pminus['Mb'], grp['x1_fit'],
                    grp['color_fit'], grp['mbfit'], pminus['Om'], pminus['w0'], pminus['wa'])

    return vva-vvb


class FitData:
    """
    class to perform a cosmo fit on data

    Parameters
    ---------------
    data: array
      data to fit

    """

    def __init__(self, data):
        # print('Number of SN for fit', len(data))
        # print set([d[name]['idr.subset'] for name in d.keys()]

        print(data.columns)
        Z = data['z_fit']
        # Mb = data['mbfit']
        Mb = -2.5*np.log10(data['x0_fit'])+10.635
        # Cov_mbmb = data['Cov_mbmb']
        Cov_mbmb = (2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']
        Cov_x1x1 = data['Cov_x1x1']
        Cov_colorcolor = data['Cov_colorcolor']
        Cov_x1mb = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        Cov_colormb = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        Cov_x1color = data['Cov_x1color']
        X1 = data['x1_fit']
        Color = data['color_fit']
        sigZ = 0.01*(1.+Z)

        # instance of the fit functions here
        self.fit = FitCosmo(Z, X1, Color, Mb, Cov_mbmb,
                            Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                            Cov_colormb, Cov_x1color, sigZ, len(data),
                            params_fit=['Om', 'w0', 'wa', 'M', 'alpha', 'beta', ])

    def __call__(self):
        """
        Method to perform the cosmological fit

        """

        # initial parameter values
        sigma_int = 0.273409368886
        Om = 0.3
        w0 = -1.0
        wa = 0.0
        M = -19.045
        M = -19.0
        alpha = 0.13
        beta = 3.1

        # first estimate of Mb
        mu = self.fit.mu(self.fit.Z, Om, w0, wa)

        mean = np.sum((self.fit.Mb-mu)/self.fit.Mber **
                      2 / np.sum(1/self.fit.Mber**2))
        print(mean)

        time_ref = time.time()

        """
        sigma_int_vals = np.arange(0.01, 0.05, 0.01)
        for sigma_int in sigma_int_vals:
            self.fit.sigma_int = sigma_int
            chii2_init = self.fit.zchii2(
                (Om, w0, wa, M, alpha, beta))-self.fit.ndf
            print('chi2 here', sigma_int, chii2_init)

        # print(test)
        sigma_int = self.fit.zfinal2()
        """
        sigma_int = 0.

       # the fit is done here
        # res = self.zfinal1(
        #    self.Mb, self.Z, self.X1, self.X2, self.sigZ)
        self.fit.sigma_int = 0.0
        """
        Omb = 0.2980310890963704
        w0b = -1.011451889936614
        wab = -0.060382299532261284
        Mbb = -18.97602852487285
        alphab = 0.12896184914584868
        betab = 3.1007710392326353
        """
        chi2 = self.fit.chi2(Om, w0, wa, M, alpha, beta)
        ndf = len(self.fit.Z)-5
        print('sigma_int and chi2', sigma_int, chi2, chi2/ndf)

        resa = self.fit.fitcosmo(Om, w0, wa, M, alpha, beta, 'minuit')
        print('fit minuit done', resa['chi2']/resa['ndf'],
              resa[['Om', 'w0', 'wa', 'M', 'alpha', 'beta']])
        """
        # print(test)
        resb = self.fit.fitcosmo(Om, w0, wa, M, alpha, beta, 'scipy')

        print('fit scipy done')
        print(resb[['Om', 'w0', 'wa', 'M', 'alpha', 'beta']])
        print(test)
        """
        """
        resa = pd.concat((resa, resb))
        resa['chi2_ndf'] = resa['chi2']/resa['ndf']
        vv = ['Om', 'w0', 'wa', 'M', 'alpha', 'beta']
        prvar = list(vv)
        for pp in vv:
            var = 'sigma_{}'.format(pp)
            resa[var] = np.sqrt(resa['Cov_{}_{}'.format(pp, pp)])
            prvar.append(var)
        prvar.append('fitter')
        prvar.append('chi2_ndf')
        ia = resa['fitter'] == 'minuit'
        ib = resa['fitter'] == 'scipy'
        print(resa[ia][prvar])
        print(resa[ib][prvar])

        print(test)
        """
        # plot Hubble diagram here
        """
        self.fit.plot_hubble(sigma_int, resa['Om'][0], resa['w0'][0],
                             wa, resa['M'][0], resa['alpha'][0], resa['beta'][0])
        """

        return resa


def plot_syste(data):

    import matplotlib.pyplot as plt
    toplot = ['sigma_mu_SN', 'sigma_bias_stat',
              'sigma_bias_x1_color', 'sigma_mu_photoz']

    for fi in data['fieldName'].unique():
        fig, ax = plt.subplots()
        fig.suptitle(fi)
        idx = data['fieldName'] == fi
        sel = data[idx].to_records(index=False)
        print('hh', fi, len(sel))
        for var in toplot:
            ax.plot(sel['z_SN'], sel[var],
                    ls='None', marker='o', label=var)
        ax.legend()
    plt.show()


class FitData_mu:
    """
    class to perform a cosmo fit on data

    Parameters
    ---------------
    data: array
      data to fit

    """

    def __init__(self, data, params_fit=['Om', 'w0'], surveyType='full'):
        # print('Number of SN for fit', len(data))
        # print set([d[name]['idr.subset'] for name in d.keys()]

        if surveyType == 'complete':
            idx = data['z_SN'] <= data['zcomp']
            data = data[idx]

        print(data.columns)
        Z_SN = data['z_SN']
        mu_SN = data['mu_SN']
        sigma_mu_SN = data['sigma_mu_SN']
        sigma_mu_bias = np.sqrt(data['sigma_bias_stat']
                                ** 2+data['sigma_bias_x1_color']**2)
        data['sigma_mu_bias'] = sigma_mu_bias
        sigma_mu_photoz = data['sigma_mu_photoz']
        self.sigma_photoz = data['sigma_photoz'].unique()[0]
        self.nsn_spectro_ultra_yearly = data['nsn_spectro_ultra_yearly'].mean()
        self.nsn_spectro_ultra_tot = data['nsn_spectro_ultra_tot'].mean()
        self.nsn_spectro_deep_yearly = data['nsn_spectro_deep_yearly'].mean()
        self.nsn_spectro_deep_tot = data['nsn_spectro_deep_tot'].mean()

        self.nsn_DD_fields = {}
        for fieldName in data['fieldName'].unique():
            if fieldName != 'WFD':
                idx = data['fieldName'] == fieldName
                sel = data[idx]
                self.nsn_DD_fields[fieldName] = len(sel)
        self.nsn_DD = len(data[data['snType'] == 'DD'])
        self.nsn_z_09 = len(data[data['z_SN'] >= 0.9])
        self.nsn_WFD = len(data[data['snType'] == 'WFD'])

        nsn_ultra_z, nsn_ultra = self.get_nsn_z(
            data, zmin=0.8, listFields=['COSMOS', 'XMM-LSS'])
        nsn_dd_z, nsn_dd = self.get_nsn_z(
            data, zmin=0.5, listFields=['CDFS', 'ADFS', 'ELAIS'])

        self.nsn_ultra_z_08 = nsn_ultra_z
        self.nsn_ultra = nsn_ultra
        self.nsn_dd_z_05 = nsn_dd_z
        self.nsn_dd = nsn_dd
        #print(nsn_ultra_z, nsn_ultra, nsn_dd_z, nsn_dd)
        print('NSN to fit', len(Z_SN))
        print('SN utra_deep', nsn_ultra)
        print('SN deep', nsn_dd)
        print('SN WFD', self.nsn_WFD)

        # instance of the fit functions here
        self.fit = FitCosmo_mu(Z_SN, mu_SN, sigma_mu_SN, sigma_mu_bias, sigma_mu_photoz,
                               params_fit=params_fit)

        # plot_syste(data)

    def __call__(self):

        Om = 0.3
        w0 = -1.0
        wa = 0.0
        self.fit.sigma_int = 0.0
        """
        chi2 = self.fit.chi2(Om, w0, wa)

        print('chi2', chi2, chi2/self.fit.ndf, self.fit.ndf)
        """
        # print('fitting sigma_int', self.fit.sigma_int)
        sigma_int = self.fit.zfinal2()
        print('sigmaInt', sigma_int)
        self.fit.sigma_int = sigma_int

        resa = self.fit.fitcosmo(Om, w0, wa)
        # print(resa.columns)
        toprint = []
        for vv in self.fit.params_fit:
            toprint.append(vv)
            toprint.append('sigma_{}'.format(vv))
        toprint.append('chi2_ndf')
        # print('fit minuit done', resa[toprint])
        resa['sigma_int'] = sigma_int
        resa['nsn_DD'] = self.nsn_DD
        resa['nsn_WFD'] = self.nsn_WFD
        resa['nsn_z_09'] = self.nsn_z_09
        resa['sigma_photoz'] = self.sigma_photoz
        resa['nsn_ultra_z_08'] = self.nsn_ultra_z_08
        resa['nsn_ultra'] = self.nsn_ultra
        resa['nsn_dd_z_05'] = self.nsn_dd_z_05
        resa['nsn_dd'] = self.nsn_dd
        resa['nsn_spectro_ultra_yearly'] = self.nsn_spectro_ultra_yearly
        resa['nsn_spectro_ultra_tot'] = self.nsn_spectro_ultra_tot
        resa['nsn_spectro_deep_yearly'] = self.nsn_spectro_deep_yearly
        resa['nsn_spectro_deep_tot'] = self.nsn_spectro_deep_tot

        for key, vals in self.nsn_DD_fields.items():
            resa['nsn_DD_{}'.format(key)] = vals

        return resa

    def get_nsn_z(self, data, zmin=0.8, listFields=['COSMOS', 'XMM-LSS']):
        """
        Method to get the number of SN with z>= zmin for a list of fields

        Parameters
        ---------------
        data: pandas df
          data to process
        zmin: float, obs
          min redshift value (default: 0.8)
        listField: list(str)
          list of fields to consider (default: COSMOS, XMM-LSS)

        Returns
        -----------
        nsn_z: number of SN with z > zmin (float)
        nsn_tot: number of sn corresponding to the fields

        """

        nsn_z = 0
        nsn_tot = 0
        idx = np.isin(data['fieldName'], listFields)
        sel = data[idx]
        if len(sel) > 0:
            idx = sel['z_SN'] >= zmin
            nsn_z = len(sel[idx])
            nsn_tot = len(sel)

        return nsn_z, nsn_tot


class FitCosmo(CosmoDist):
    """
    Class to fit cosmology from a set of data

    Parameters
    ---------------

    """

    def __init__(self, Z, X1, Color, Mb, Cov_mbmb,
                 Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                 Cov_colormb, Cov_x1color, sigZ, nsn,
                 params_fit=['Om', 'w0', 'wa', 'M', 'alpha', 'beta'],
                 H0=72, c=299792.458):
        super().__init__(H0, c)

        # print('Number of SN for fit', nsn)
        # print set([d[name]['idr.subset'] for name in d.keys()]

        self.Z = Z
        self.Mb = Mb
        self.Mber = np.sqrt(Cov_mbmb)
        self.gx1 = Cov_x1x1
        self.gxc = Cov_colorcolor
        self.cov1 = Cov_x1mb
        self.cov2 = Cov_colormb
        self.cov3 = Cov_x1color
        self.sigZ = sigZ
        self.X1 = X1
        self.Color = Color
        self.ndf = nsn-len(params_fit)-1
        self.params_fit = params_fit

    def plot_modulus(self, offset=0):
        z = np.arange(0.001, 0.12, 0.001)
        for Ol in [0, 0.3, 0.7, 1]:
            # hodl=luminosity_distance(z,Ol)
            r = self.mu(z, Ol)
            plt.plot(z, hodl)
            plt.plot(z, r+offset, 'x')
        plt.show()

    def sigmI(self, alpha, beta):
        """
        Method to estimate the variance of mu from SN parameters
        mu = Mb-M+alpha*x1-beta*color

        Parameters
        ---------------
        alpha: float
          alpha parameter(for SN standardization)
        beta: float
          beta parameter

        Returns
        ----------
        var_mu = Mberr**2+alpha**2*sigma_x1+beta**2*sigma_color+2 * \
            alpha*Cov_x1_mb-2*alpha*beta*Cov_x1_color-2*beta*Cov_mb_color
        """
        res = self.Mber**2+(alpha**2)*self.gx1+(beta**2)*self.gxc + \
            2*alpha*self.cov1-2*beta*self.cov2-2*alpha*beta*self.cov3
        """
        print('hhh', np.sqrt(self.gx1), np.sqrt(self.gxc))
        import matplotlib.pyplot as plt
        plt.hist(alpha**2*self.gx1, histtype='step', label='gx1')
        plt.hist(beta**2*self.gxc, histtype='step', label='gcx')
        plt.hist(2*alpha*self.cov1, histtype='step', label='cov1')
        plt.hist(-2*beta*self.cov2, histtype='step', label='cov2')
        plt.hist(-2*alpha*beta*self.cov3, histtype='step', label='cov3')
        plt.legend()
        plt.show()
        """
        return res

    def zchii2(self, tup):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit(Om, w0, wa, M, alpha, beta)

        Returns
        ----------
        the cosmo chi2
        """
        Om, w0, wa, M, alpha, beta = tup
        # return np.sum((Mb-self.mu(Z, Om, w0, wa)-M+alpha*x1-beta*color)**2/(self.sigmI(alpha, beta)+sigma_int**2+self.sigMu(1.-Om, Z, sigZ)**2))
        # return np.sum((self.Mb-self.mu(self.Z, Om, w0, wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+self.sigma_int**2+self.sigZ**2))
        return self.chi2(Om, w0, wa, M, alpha, beta)

    def zchii2_nowa(self, tup):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit(Om, w0, M, alpha, beta)
         wa: float
           wa parameter
         sigma_int: float
           sigma_int parameter

        Returns
        ----------
        the cosmo chi2

        """
        Om, w0, M, alpha, beta = tup

        # return np.sum((self.Mb-self.mu(self.Z, Om, w0, self.wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+self.sigma_int**2+self.sigMu(Om, w0, self.wa)**2))
        return self.chi2(Om, w0, self.wa, M, alpha, beta)

    def chi2(self, Om, w0, wa, M, alpha, beta):

        # self.sigMu(Om, w0, wa)**2))
        """
        print('dl', self.mu(self.Z, Om, w0, wa),
              self.mu_old(self.Z, Om, w0, wa))
        """
        return np.sum((self.Mb-M+alpha*self.X1-beta*self.Color-self.mu(self.Z, Om, w0, wa))**2/(self.sigmI(alpha, beta)+self.sigma_int**2))

    def zchii2_nowa_im(self, Om, w0, M, alpha, beta):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit (Om, w0, M, alpha, beta)
         wa: float
           wa parameter
         sigma_int: float
           sigma_int parameter

        Returns
        ----------
        the cosmo chi2

        """
        # Om, w0, M, alpha, beta = tup
        # return np.sum((self.Mb-self.mu(self.Z, Om, w0, self.wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+self.sigma_int**2+self.sigMu(Om, w0, self.wa)**2))
        return self.chi2(Om, w0, self.wa, M, alpha, beta)

    def zchii2_im(self, Om, w0, wa, M, alpha, beta):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit (Om, w0, M, alpha, beta)
         wa: float
           wa parameter
         sigma_int: float
           sigma_int parameter

        Returns
        ----------
        the cosmo chi2

        """
        # Om, w0, M, alpha, beta = tup
        # return np.sum((self.Mb-self.mu(self.Z, Om, w0, wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+self.sigma_int**2+self.sigMu(Om, w0, wa)**2))

        res = self.chi2(Om, w0, wa, M, alpha, beta)
        return res

    def zfinal1(self, sigma_int):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0, wa
        """
        self.sigma_int = sigma_int
        return optimize.minimize(self.zchii2, (0.3, -1.0, 0., -19., 0.13, 3.1))

    def fitcosmo(self, Om, w0, wa, M, alpha, beta, fitter='minuit'):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0

        Parameters
        --------------
        wa: float
          wa DE parameter

        Returns
        ----------
        fitted parameters

        """

        if fitter == 'minuit':
            return self.fitcosmo_minuit(Om, w0, wa, M, alpha, beta)

        if fitter == 'scipy':
            return self.fitcosmo_scipy(Om, w0, wa, M, alpha, beta)

    def fitcosmo_minuit(self, Om, w0, wa, M, alpha, beta):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0, and wa

        Parameters
        --------------
       Om, w0, wa, M, alpha, beta: float
          cosmo parameters to fit

        Returns
        ----------
        fitted parameters

        """
        if 'wa' not in self.params_fit:
            self.wa = wa
            m = Minuit(self.zchii2_nowa_im, Om=Om, w0=w0, M=M,
                       alpha=alpha, beta=beta)
        else:
            m = Minuit(self.zchii2_im, Om=Om, w0=w0, wa=wa, M=M,
                       alpha=alpha, beta=beta)

        # perform the fit here
        dictout = {}
        try:
            m.migrad()
            values = m.values

            for key, vals in values.items():
                dictout[key] = [vals]
                m.hesse()   # run covariance estimator
            if m.covariance is not None:
                for key, vals in m.covariance.items():
                    what = '{}_{}'.format(key[0], key[1])
                    dictout['Cov_{}'.format(what)] = [vals]

                dictout['accuracy'] = [m.accurate]
                dictout['chi2'] = [m.fval]
                dictout['ndf'] = [self.ndf]
                dictout['fitter'] = ['minuit']
            if 'wa' not in self.params_fit:
                dictout['wa'] = [wa]
        except (RuntimeError, TypeError, NameError):
            print('Fit crash')

        return pd.DataFrame.from_dict(dictout)

    def fitcosmo_scipy(self, Om, w0, wa, M, alpha, beta):

        if 'wa' not in self.params_fit:
            self.wa = wa
            res = optimize.minimize(
                self.zchii2_im, (Om, w0, M, alpha, beta))
        else:
            res = optimize.minimize(
                self.zchii2, (Om, w0, wa, M, alpha, beta))

        print(res.x)
        # get the covariance matrix
        covmat = res.hess_inv
        # covmat = res.jac
        print(np.sqrt(np.diag(covmat)))

        # Extract the parameters and copy in a dict
        params = {}

        for i, vv in enumerate(self.params_fit):
            params[vv] = [res.x[i]]
            for j, pp in enumerate(self.params_fit):
                if j >= i:
                    nn = 'Cov_{}_{}'.format(vv, pp)
                    params[nn] = [covmat[i][j]]

        tup = []
        for pp in self.params_fit:
            tup.append(params[pp][0])

        # chi2 = self.zchii2(tuple(tup), self.Mb, self.Z, self.X1, self.X2, self.sigZ)/self.ndf
        if 'wa' not in self.params_fit:
            chi2 = self.zchii2_nowa(tuple(tup))
        else:
            chi2 = self.zchii2(tuple(tup))
        print('hhh', tup, chi2)
        params['chi2'] = [chi2]
        params['ndf'] = [self.ndf]
        params['accuracy'] = [1]
        params['fitter'] = ['scipy']
        return pd.DataFrame.from_dict(params)

    def zchi2ndf(self, sigma_int):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        sigma_int: float
          sigma_int param

        Returns
        ----------
        fitted parameters
        """
        res = self.zfinal1(sigma_int)
        om, w0,  wa, m, xi1, xi2 = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5]
        print('jjj', om, w0, wa, m, xi1, xi2, sigma_int)
        return self.zchii2((om, w0, wa, m, xi1, xi2))-self.ndf

    def zfinal2(self):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Returns
        ----------
        sigma_int parameter
         """
        return optimize.newton(self.zchi2ndf, 0.01)

    def zchi2ndf_nowa(self, sigma_int, wa):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        wa: float
          wa cosmo parameter
        sigma_int: float
          sigma_int param

        Returns
        ----------
        fitted parameters
        """
        res = self.zfinal1_nowa(sigma_int, wa)
        om, w0,  m, xi1, xi2 = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]
        # om, w0,  m, xi1, xi2 = 0.3, -1.0, -19.045, 0.13, 3.1
        print('jjj', om, w0,  m, xi1, xi2, sigma_int, self.ndf)
        return self.zchii2_nowa((om, w0, m, xi1, xi2), wa, sigma_int)-self.ndf

    def zfinal2_nowa(self, wa):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        wa : float
          wa cosmo parameter

        Returns
        ----------
        sigma_int parameter
        """
        return optimize.newton(self.zchi2ndf_nowa, 0.00615, args=(wa,))

    def derivate2(self, Om, w0, wa):
        """
        Method to estimate the derivative of the distance modulus wrt redshift

        Parameters
        ---------------
        Om: float
          Omega_m cosmo parameter
        w0: float
          w0 cosmo parameter
        wa: float
          wa cosmo parameter

        Returns
        ----------
        The dertivative of the distance modulus wrt redshift
        """
        dl = self.dL(self.Z, Om, w0, wa)
        integrand = self.cosmo_func(self.Z, Om, w0, wa)
        coeff = 5./np.log(10)
        val1 = coeff/(1+self.Z)
        val2 = coeff*(1+self.Z)*integrand/dl
        norm = self.c/self.H0
        norm *= 1.e6
        return val1+norm*val2

    def sigMu(self, Om, w0, wa):
        """
        Method to estimate the error on the distance modulus
        due to the redshift measurement error

        Parameters
        --------------
        Om: float
          Omega_m cosmo parameter
        w0: float
          w0 cosmo parameter
        wa: float
          wa cosmo parameter

        Returns
        ----------
        sigma_mu due to redshift mezsurement error

        """
        return self.derivate2(Om, w0, wa)*self.sigZ

    def plot_hubble(self, sigma_int, Om, w0, wa, M, alpha, beta):
        """
        Method to perform a Hubble plot

        Parameters
        --------------
        sigma_int: float
          sigma_int parameter
        Om: float
          Omega_m cosmo parameter
        w0: float
          w0 cosmo parameter
        wa: float
          wa cosmo parameter
        M, alpha, beta: float
          nuisance parameters
        """
        from . import plt
        plt.errorbar(self.Z, self.Mb+alpha*self.X1-beta*self.Color,
                     yerr=np.sqrt(self.sigmI(alpha, beta)+sigma_int**2+self.sigMu(Om, w0, wa)**2), xerr=None, fmt='o')
        z = np.arange(0.001, 1., 0.001)
        r = self.mu(z, Om, w0, wa)
        plt.plot(z, r+M, 'x')
        plt.show()


class FitCosmo_mu(CosmoDist):
    """
    Class to fit cosmology from a set of data

    Parameters
    ---------------

    """

    def __init__(self, Z, mu, sigma_mu, sigma_bias, sigma_mu_photoz,
                 params_fit=['Om', 'w0'],
                 H0=70., c=299792.458):
        super().__init__(H0, c)

        nsn = len(Z)
        # print('Number of SN for fit', nsn)

        self.Z_SN = Z
        self.mu_SN = mu
        self.sigma_mu_SN = sigma_mu
        self.sigma_bias = sigma_bias
        self.sigma_mu_photoz = sigma_mu_photoz
        self.ndf = nsn-len(params_fit)-1
        self.params_fit = params_fit

    def plot_modulus(self, offset=0):
        z = np.arange(0.001, 0.12, 0.001)
        for Ol in [0, 0.3, 0.7, 1]:
            # hodl=luminosity_distance(z,Ol)
            r = self.mu(z, Ol)
            plt.plot(z, hodl)
            plt.plot(z, r+offset, 'x')
        plt.show()

    def fitcosmo(self, Om, w0, wa):
        """
        Method to estimate cosmo fit parameters Om, w0, and wa

        Parameters
        --------------
       Om, w0, wa : float
          cosmo parameters to fit

        Returns
        ----------
        fitted parameters

        """

        # perform the fit here
        df = pd.DataFrame()
        try:
            if 'wa' not in self.params_fit:
                self.wa = wa
                m = Minuit(self.chi2_nowa, Om=Om, w0=w0)
            else:
                m = Minuit(self.chi2, Om=Om, w0=w0, wa=wa)
            m.migrad()
            dictout = {}
            values = m.values
            for key, vals in values.items():
                dictout[key] = [vals]
            m.hesse()   # run covariance estimator
            if m.covariance is not None:
                for key, vals in m.covariance.items():
                    what = '{}_{}'.format(key[0], key[1])
                    dictout['Cov_{}'.format(what)] = [vals]

            dictout['accuracy'] = [m.accurate]
            dictout['chi2'] = [m.fval]
            dictout['ndf'] = [self.ndf]

            dictout['fitter'] = ['minuit']
            if 'wa' not in self.params_fit:
                dictout['wa'] = [wa]
            for vv in self.params_fit:
                vvb = 'Cov_{}_{}'.format(vv, vv)
                if vvb in list(dictout.keys()):
                    dictout['sigma_{}'.format(vv)] = np.sqrt(dictout[vvb])

            df = pd.DataFrame.from_dict(dictout)
            df['chi2_ndf'] = df['chi2']/df['ndf']
        except (RuntimeError, TypeError, NameError, ValueError):
            print('Fit crash')

        return df

    def chi2ndf(self, sigma_int):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        sigma_int: float
          sigma_int param

        Returns
        ----------
        fitted parameters
        """
        res = self.zfinal1(sigma_int)
        om, w0,  wa = res.x[0], res.x[1], res.x[2]

        res = self.tchi2((om, w0, wa, sigma_int))-self.ndf
        return res

    def chi2ndf_sigma_int(self, sigma_int):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        sigma_int: float
          sigma_int param

        Returns
        ----------
        fitted parameters
        """
        om, w0,  wa = 0.3, -1.0, 0.0

        res = self.chi2_sigma_int(om, w0, wa, sigma_int)-self.ndf
        return res

    def zfinal2(self):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Returns
        ----------
        sigma_int parameter
         """
        return optimize.newton(self.chi2ndf_sigma_int, 0.005)

    def zfinal1(self, sigma_int):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0, wa
        """
        return optimize.minimize(self.tchi2, (0.3, -1.0, 0., sigma_int))
        # return optimize.minimize(self.tchi2, (0.3, -1.0, 0., sigma_int))

    def chi2(self, Om, w0, wa):
        return(np.sum((self.mu_SN-self.mu_astro(self.Z_SN, Om, w0, wa))**2/(self.sigma_mu_SN**2+self.sigma_int**2+self.sigma_bias**2+self.sigma_mu_photoz**2)))

    def chi2_sigma_int(self, Om, w0, wa, sigma_int):

        return np.sum((self.mu_SN-self.mu_astro(self.Z_SN, Om, w0, wa))**2/(self.sigma_mu_SN**2+sigma_int**2+self.sigma_bias**2+self.sigma_mu_photoz**2))

    def chi2_nowa(self, Om, w0):

        return(np.sum((self.mu_SN-self.mu_astro(self.Z_SN, Om, w0, self.wa))**2/(self.sigma_mu_SN**2+self.sigma_int**2+self.sigma_bias**2+self.sigma_mu_photoz**2)))

    def tchi2(self, tup):
        Om, w0, wa, sigma_int = tup

        rr = self.chi2_sigma_int(Om, w0, wa, sigma_int)
        return rr


class Sigma_Fisher(CosmoDist):
    """"
    class to estimate error parameters using Fisher matrices

    """

    def __init__(self, data,
                 params=dict(zip(['M', 'alpha', 'beta', 'Om', 'w0',
                                  'wa'], [-19.045, 0.13, 2.96, 0.3, -1.0, 0.0])),
                 params_Fisher=['Om', 'w0', 'wa', 'M', 'alpha', 'beta', ],
                 H0=72, c=299792.458):
        super().__init__(H0, c)

        Z = data['z_fit']
        Mb = -2.5*np.log10(data['x0_fit'])+10.635
        # Cov_mbmb = data['Cov_mbmb']
        Cov_mbmb = (2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']
        Cov_x1x1 = data['Cov_x1x1']
        Cov_colorcolor = data['Cov_colorcolor']
        Cov_x1mb = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        Cov_colormb = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        Cov_x1color = data['Cov_x1color']
        X1 = data['x1_fit']
        Color = data['color_fit']
        sigZ = 0.01*(1.+Z)
        # sigZ = 0.0
        # instance of the fit functions here
        self.fit = FitCosmo(Z, X1, Color, Mb, Cov_mbmb,
                            Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                            Cov_colormb, Cov_x1color, sigZ, len(data),
                            params_fit=['M', 'alpha', 'beta', 'Om', 'w0'])
        # fit parameters
        self.params = params
        self.params_Fisher = params_Fisher
        self.data = data
        self.Om = params['Om']
        self.w0 = params['w0']
        self.wa = params['wa']
        self.M = params['M']
        self.alpha = params['alpha']
        self.beta = params['beta']
        data['Mb'] = -2.5*np.log10(data['x0_fit'])+10.635
        data['Cov_mb_mb'] = (
            2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']
        data['Cov_x1mb'] = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        data['Cov_colormb'] = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        data['sigma_mu'] = self.data.apply(
            lambda x: self.sigma_mu(x, self.alpha, self.beta), axis=1)

        self.data = data

    def __call__(self):
        """
        paramtest = {}
        paramtest['a'] = 0.1
        paramtest['b'] = 3.

        self.datat = self.datatest(paramtest)
        print('chi2',self.chi2test(paramtest))
        print('derivative a')
        deriva = self.derivative2_same(self.chi2test,paramtest,'a')
        print('derivative b')
        derivb = self.derivative2_same(self.chi2test,paramtest,'b')
        print('derivative ab')
        derivab = self.derivative2_mix(self.chi2test,paramtest,'a','b')
        print(deriva,self.derivtest_a(paramtest),
              derivb, self.derivtest_b(paramtest),
              derivab, self.derivtest_ab(paramtest))
        print(test)
        """
        # get covariance matrix (from Hessian)
        res = self.matCov()
        # resb = self.Hessian()

        print(res)
        if res is not None:
            return dict(zip(self.params_Fisher, [res[i] for i in range(len(res))]))
        else:
            return dict(zip(self.params_Fisher, [-999]*len(self.params_Fisher)))

    def matCov(self):

        parName = self.params_Fisher
        # parName = ['Om','w0','wa']
        nparams = len(parName)

        # get derivatives
        # deriv = self.derivative_chi2(parName)
        Fisher_Matrix = np.zeros((nparams, nparams))

        self.data['chi2_indiv'] = self.data.apply(
            lambda x: self.chi2_indiv(x, self.params), axis=1)
        for pp in parName:
            self.data['d_{}'.format(pp)] = self.data.apply(
                lambda x: self.derivative_grp(x, self.derivative_Fisher, self.params, pp), axis=1)

        # construct the Jacobian matrix
        ndata = len(self.data)
        J = np.zeros((ndata, nparams))

        print('data', ndata, nparams)
        io = -1
        for i, row in self.data.iterrows():
            io += 1
            for j, pp in enumerate(parName):
                J[io, j] = row['d_{}'.format(pp)]

        # second order derivatives
        """
        hpar = dict(zip(parName, [1.e-1, 1.e-1, 1.e-1, 1.e-1, 1.e-3, 1.e-3]))
        for j, pp in enumerate(parName):
            for jb, ppb in enumerate(parName):
                what = 'd2_{}_{}'.format(pp, ppb)
                if pp == ppb:
                    self.data[what] = self.data.apply(lambda x: self.derivative2_same(x,
                                                                                      self.chi2_indiv, self.params, pp, h=hpar[pp]), axis=1)
                else:
                    self.data[what] = self.data.apply(lambda x: self.derivative2_mix(x,
                                                                                     self.chi2_indiv, self.params, pp, ppb, ha=hpar[pp], hb=hpar[ppb]), axis=1)

        Hsec = np.zeros((nparams, nparams))
        for i, pp in enumerate(parName):
            for j, ppb in enumerate(parName):
                Hsec[i, j] = np.sum(self.data['chi2_indiv']
                                    * self.data['d2_{}_{}'.format(pp, ppb)])
        """
        # Hessian
        # H = 2.*(np.dot(J.T, J)+Hsec)
        H = 2*np.dot(J.T, J)
        detmat = np.linalg.det(H)
        Cov = None
        if detmat:
            H_inv = np.linalg.inv(H)
            Cov = np.sqrt(np.diag(H_inv))

        return Cov

    def Hessian(self):

        parName = self.params_Fisher
        # parName = ['Om','w0','wa']
        nparams = len(parName)

        # get derivatives
        # deriv = self.derivative_chi2(parName)
        # fill the Fisher matrix
        ndata = len(self.data)

        for j, pp in enumerate(parName):
            for jb, ppb in enumerate(parName):
                what = 'd2_{}_{}'.format(pp, ppb)
                if pp == ppb:
                    self.data[what] = self.data.apply(lambda x: self.derivative2_same(x,
                                                                                      self.chi2_indiv, self.params, pp), axis=1)
                else:
                    self.data[what] = self.data.apply(lambda x: self.derivative2_mix(x,
                                                                                     self.chi2_indiv, self.params, pp, ppb), axis=1)

        H = np.zeros((nparams, nparams))
        io = -1
        for i, pp in enumerate(parName):
            for j, ppb in enumerate(parName):
                H[i, j] = np.sum(self.data['d2_{}_{}'.format(pp, ppb)])

        print(H)
        Cov = None
        detmat = np.linalg.det(H)
        if detmat:
            H_inv = np.linalg.inv(H)
            Cov = np.sqrt(np.diag(H_inv))
        return Cov

    def derivative_Fisher(self, grp, params, parName, h=1.e-8):
        pars = {}
        for key, pp in params.items():
            pars[key] = pp + h*(parName == key)

        Om = pars['Om']
        w0 = pars['w0']
        wa = pars['wa']
        M = pars['M']
        alpha = pars['alpha']
        beta = pars['beta']

        sigma_int = 0.011
        res = (grp.Mb-M+alpha*grp.x1-beta*grp.color-self.mu(grp.z, Om,
                                                            w0, wa))**2/(self.sigma_mu(grp, alpha, beta)**2+sigma_int**2)
        return np.sqrt(res)

    def derivative_grp(self, grp, func, params, parName, h=1.e-8):

        deriv = (func(grp, params, parName, h) -
                 func(grp, params, parName, -h))/(2*h)

        return deriv

    def derivative(self, func, params, parName, h=1.e-8):

        deriv = (func(params, parName, h)-func(params, parName, -h))/(2*h)

        return deriv

    def derivative2_same(self, grp, func, params, parName, h=1.e-8):

        deriv = (func(grp, params, parName, h)-2.*func(grp, params) +
                 func(grp, params, parName, -h))/(h**2)

        return deriv

    def derivative2_mix(self, grp, func, params, parName1, parName2, ha=1.e-8, hb=1.e-8):

        deriv = func(grp, params, parName1, ha, parName2, hb)
        deriv -= func(grp, params, parName1, -ha, parName2, hb)
        deriv -= func(grp, params, parName1, ha, parName2, -hb)
        deriv += func(grp, params, parName1, -ha, parName2, -hb)
        deriv /= 4*ha*hb

        return deriv

    def chi2_indiv(self, grp, param, parNamea='', ha=0, parNameb='', hb=0):

        pars = {}
        for key, pp in param.items():
            pars[key] = pp
            if key == parNamea:
                pars[key] += ha
            if key == parNameb:
                pars[key] += hb

        Om = pars['Om']
        w0 = pars['w0']
        wa = pars['wa']
        M = pars['M']
        alpha = pars['alpha']
        beta = pars['beta']

        sigma_int = 0.011
        res = (grp.Mb-M+alpha*grp.x1-beta*grp.color-self.mu(grp.z, Om,
                                                            w0, wa))**2/(self.sigma_mu(grp, alpha, beta)**2+sigma_int**2)
        return np.sqrt(res)

    def datatest(self, params, sigma=0.5):

        x = np.arange(0., 30, 0.5)
        y = self.functest(params, x)
        data = np.random.randn(len(y)) * sigma + y

        df = pd.DataFrame(x, columns=['x'])
        df['y'] = y
        df['sigma'] = sigma
        return df

    def functest(self, params, x):
        # print('ffun',params)
        y = params['a']*x**3+params['b']*x**2+params['a']*params['b']*x
        return y

    def chi2test(self, params, parNamea='', ha=0, parNameb='', hb=0):

        pars = {}
        for key, pp in params.items():
            pars[key] = pp + ha*(parNamea == key)+hb*(parNameb == key)

        print('pp', pars)
        return np.sum((self.datat['y']-self.functest(pars, self.datat['x']))**2/self.datat['sigma']**2)

    def derivtest_a(self, params):

        return 2.*np.sum((self.datat['x']**3+params['b']*self.datat['x'])**2/self.datat['sigma']**2)

    def derivtest_b(self, params):

        return 2.*np.sum((self.datat['x']**2+params['a']*self.datat['x'])**2/self.datat['sigma']**2)

    def derivtest_ab(self, params):
        suma = -np.sum((self.datat['x']**2+params['a']*self.datat['x'])*(
            self.datat['x']**3+params['b']*self.datat['x'])/self.datat['sigma']**2)
        sumb = np.sum((self.datat['y']-self.functest(params,
                                                     self.datat['x']))*self.datat['x']/self.datat['sigma']**2)

        return -2.*(suma+sumb)

    def sigma_mu(self, grp, alpha, beta):

        res = grp.Cov_mbmb+(alpha**2)*grp.Cov_x1x1+(beta**2)*grp.Cov_colorcolor + \
            2*alpha*grp.Cov_x1mb-2*beta*grp.Cov_colormb-2*alpha*beta*grp.Cov_x1color
        return np.sqrt(res)


class Sigma_Fisher_mu(CosmoDist):
    """"
    class to estimate error parameters using Fisher matrices

    """

    def __init__(self, data,
                 params=dict(zip(['Om', 'w0',
                                  'wa'], [0.3, -1.0, 0.0])),
                 params_Fisher=['Om', 'w0', 'wa'],
                 H0=70, c=299792.458):
        super().__init__(H0, c)

        self.data = data
        # fit parameters
        self.params = params
        self.ndf = len(self.data)-len(params_Fisher)-1
        self.params_Fisher = params_Fisher

        self.z_SN = data['z_SN']
        self.mu_SN = data['mu_SN']
        self.sigma_mu_SN = data['sigma_mu_SN']
        self.sigma_bias = data['sigma_bias']
        # need to estimate sigma_int
        self.sigma_int = self.get_sigma_int()
        print('sigma_int', self.sigma_int)
        print('chisquare', self.chi2(
            params['Om'], params['w0'], params['wa'])/self.ndf)

    def __call__(self):

        # get covariance matrix (from Hessian)
        res = self.matCov()
        # resb = self.Hessian()

        # print(res, resb)
        if res is not None:
            return dict(zip(self.params_Fisher, [res[i] for i in range(len(res))]))
        else:
            return dict(zip(self.params_Fisher, [-999]*len(self.params_Fisher)))

    def matCov(self):

        parName = self.params_Fisher
        nparams = len(parName)

        # get derivatives
        # deriv = self.derivative_chi2(parName)
        Fisher_Matrix = np.zeros((nparams, nparams))

        self.data['chi2_indiv'] = self.data.apply(
            lambda x: self.chi2_indiv(x, self.params), axis=1)
        for pp in parName:
            self.data['d_{}'.format(pp)] = self.data.apply(
                lambda x: self.derivative_grp(x, self.derivative_Fisher, self.params, pp), axis=1)

        # construct the Jacobian matrix
        ndata = len(self.data)
        J = np.zeros((ndata, nparams))

        io = -1
        for i, row in self.data.iterrows():
            io += 1
            for j, pp in enumerate(parName):
                J[io, j] = row['d_{}'.format(pp)]

        # Fisher matrix
        FisherMat = np.dot(J.T, J)
        detmat = np.linalg.det(FisherMat)
        Cov = None
        if detmat:
            F_inv = np.linalg.inv(FisherMat)
            Cov = np.sqrt(np.diag(F_inv))

        return Cov

    def derivative_Fisher(self, grp, params, parName, h=1.e-8):
        pars = {}
        for key, pp in params.items():
            pars[key] = pp + h*(parName == key)

        Om = pars['Om']
        w0 = pars['w0']
        wa = pars['wa']

        res = (grp.mu_SN-self.mu(grp.z_SN, Om, w0, wa))**2 / \
            (grp.sigma_mu_SN**2+self.sigma_int**2)
        return np.sqrt(res)

    def derivative_grp(self, grp, func, params, parName, h=1.e-8):

        deriv = (func(grp, params, parName, h) -
                 func(grp, params, parName, -h))/(2*h)

        return deriv

    def chi2_indiv(self, grp, param, parNamea='', ha=0, parNameb='', hb=0):

        pars = {}
        for key, pp in param.items():
            pars[key] = pp
            if key == parNamea:
                pars[key] += ha
            if key == parNameb:
                pars[key] += hb

        Om = pars['Om']
        w0 = pars['w0']
        wa = pars['wa']

        res = (grp.mu_SN-self.mu_astro(grp.z_SN, Om, w0, wa))**2 / \
            (grp.sigma_mu_SN**2+self.sigma_int**2)
        return np.sqrt(res)

    def chi2(self, Om, w0, wa):

        res = np.sum((self.mu_SN-self.mu_astro(self.z_SN, Om, w0, wa))**2 /
                     (self.sigma_mu_SN**2+self.sigma_int**2))
        return res

    def chi2ndf_sigma_int(self, sigma_int):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        sigma_int: float
          sigma_int param

        Returns
        ----------
        fitted parameters
        """
        Om, w0,  wa = 0.3, -1.0, 0.0
        rr = np.sum((self.mu_SN-self.mu_astro(self.z_SN, Om, w0, wa))
                    ** 2/(self.sigma_mu_SN**2+sigma_int**2))
        res = rr-self.ndf
        return res

    def get_sigma_int(self):
        """
        Method to estimate the sigma_int parameter
        to get a chisquare equal to 1

        Returns
        ----------
        sigma_int parameter
         """
        return optimize.newton(self.chi2ndf_sigma_int, 0.01)
