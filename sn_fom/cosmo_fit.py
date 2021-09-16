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

    def __init__(self, H0=72, c=2.99792e5):

        self.H0 = H0
        self.c = c

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
        #wp = w0

        H = Om*(1+z)**3+(1.-Om)*(1+z)**(3*(1.+wp))
        # H = Om*(1+z)**3+(1.-Om)*(1+z)

        fu = np.sqrt(H)

        return 1/fu

    def dL(self, z, Om=0.3, w0=-1., wa=0.0):
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

        def integrand(x): return norm*self.cosmo_func(x, Om, w0, wa)

        if (hasattr(z, '__iter__')):
            s = np.zeros(len(z))
            for i, t in enumerate(z):
                s[i] = (1+t)*quad(integrand, 0, t, limit=100)[0]
            return s
        else:
            return (1+z)*quad(integrand, 0, z, limit=100)[0]

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

        # return 5.*np.log10(self.dL(z, Om, w0, wa))+25. #if dL in Mpc

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
        print('Number of SN', len(self.data))
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
        print('Number of SN for fit', len(data))
        # print set([d[name]['idr.subset'] for name in d.keys()]

        print(data.columns)
        Z = data['z_fit']
        Mb = data['mbfit']
        Cov_mbmb = data['Cov_mbmb']
        Cov_x1x1 = data['Cov_x1x1']
        Cov_colorcolor = data['Cov_colorcolor']
        Cov_x1mb = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        Cov_x1mb = data['Cov_x1mb']
        Cov_colormb = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        Cov_colormb = data['Cov_colormb']
        Cov_x1color = data['Cov_x1color']
        X1 = data['x1_fit']
        Color = data['color_fit']
        sigZ = 0.01*(1.+Z)

        # instance of the fit functions here
        self.fit = FitCosmo(Z, X1, Color, Mb, Cov_mbmb,
                            Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                            Cov_colormb, Cov_x1color, sigZ, len(data),
                            params_fit=['M', 'alpha', 'beta', 'Om', 'w0'])

    def __call__(self):
        """
        Method to perform the cosmological fit

        """

        # initial parameter values
        gzero = 0.273409368886
        Om = 0.3
        w0 = -1.0
        wa = 0.0
        m = -19.045
        alpha = 0.13
        beta = 3.1

        # first estimate of Mb
        mu = self.fit.mu(self.fit.Z, Om, w0, wa)

        mean = np.sum((self.fit.Mb-mu)/self.fit.Mber **
                      2 / np.sum(1/self.fit.Mber**2))
        print(mean)

        time_ref = time.time()
        """
        gzero_vals = np.arange(0.0001,0.0005,0.00001)
        for gzero in gzero_vals:
            chii2_init = self.fit.zchii2_nowa((Om, w0, m, alpha,beta), wa,gzero)-self.fit.ndf
            print('chi2 here',gzero, chii2_init)

        print(test)
        #gzero=self.fit.zfinal2_nowa(wa)
        """
        gzero = 0.
        print('gzero', gzero)
       # the fit is done here
        # res = self.zfinal1(
        #    self.Mb, self.Z, self.X1, self.X2, self.sigZ)
        self.fit.wa = wa
        self.fit.gzero = gzero
        res = self.fit.zfinal1_nowa()
        print('after fit', time.time()-time_ref)
        print(res.x)
        # get the covariance matrix
        covmat = res.hess_inv
        #covmat = res.jac
        print(np.sqrt(np.diag(covmat)))

        # Extract the parameters and copy in a dict
        params = {}

        parNames = ['Om', 'w0', 'wa', 'M', 'alpha', 'beta']
        parNames = ['Om', 'w0', 'M', 'alpha', 'beta']

        for i, vv in enumerate(parNames):
            params[vv] = [res.x[i]]
            for j, pp in enumerate(parNames):
                if j >= i:
                    nn = 'Cov_{}_{}'.format(vv, pp)
                    params[nn] = [covmat[i][j]]

        tup = []
        for pp in parNames:
            tup.append(params[pp][0])

        #chi2 = self.zchii2(tuple(tup), self.Mb, self.Z, self.X1, self.X2, self.sigZ)/self.ndf
        chi2 = self.fit.zchii2_nowa(tuple(tup), wa, gzero)/self.fit.ndf
        print('hhh', tup, chi2)
        params['chi2'] = [chi2]

        fit_result = pd.DataFrame.from_dict(params)

        # plot Hubble diagram here
        #self.fit.plot_hubble(gzero, params['Om'][0], params['w0'][0], wa, params['M'][0], params['alpha'][0], params['beta'][0])

        return fit_result


class FitCosmo(CosmoDist):
    """
    Class to fit cosmology from a set of data

    Parameters
    ---------------


    """

    def __init__(self, Z, X1, Color, Mb, Cov_mbmb,
                 Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                 Cov_colormb, Cov_x1color, sigZ, nsn,
                 params_fit=['M', 'alpha', 'beta', 'Om', 'w0'],
                 H0=72, c=299792.458):
        super().__init__(H0, c)

        print('Number of SN for fit', nsn)
        # print set([d[name]['idr.subset'] for name in d.keys()]

        self.Z = Z
        self.Mb = Mb
        self.Mber = np.sqrt(Cov_mbmb)
        self.gx1 = Cov_x1x1
        self.gxc = Cov_colorcolor
        # self.cov1 = -2.5*data['Cov_x0x1'] / \
        #    (data['x0_fit']*np.log(10))
        self.cov1 = Cov_x1mb
        # self.cov2 = -2.5*data['Cov_x0color'] / \
        #    (data['x0_fit']*np.log(10))
        self.cov2 = Cov_colormb
        self.cov3 = Cov_x1color
        self.sigZ = sigZ
        self.X1 = X1
        self.Color = Color
        self.ndf = nsn-len(params_fit)

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
          alpha parameter (for SN standardization)
        beta: float
          beta parameter

        Returns
        ----------
        var_mu = Mberr**2+alpha**2*sigma_x1+beta**2*sigma_color+2*alpha*Cov_x1_mb-2*alpha*beta*Cov_x1_color-2*beta*Cov_mb_color
        """
        return self.Mber**2+(alpha**2)*self.gx1+(beta**2)*self.gxc+2*alpha*self.cov1-2*beta*self.cov2-2*alpha*beta*self.cov3

    def zchii2(self, tup, gzero):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit (Om, w0, wa, M, alpha, beta)

        Returns
        ----------
        the cosmo chi2
        """
        Om, w0, wa, M, alpha, beta = tup
        # return np.sum((Mb-self.mu(Z, Om, w0, wa)-M+alpha*x1-beta*color)**2/(self.sigmI(alpha, beta)+gzero**2+self.sigMu(1.-Om, Z, sigZ)**2))
        return np.sum((self.Mb-self.mu(Z, Om, w0, wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+gzero**2+self.sigZ**2))

    def zchii2_nowa(self, tup, wa, gzero):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit (Om, w0, M, alpha, beta)
         wa: float
           wa parameter
         gzero: float
           gzero parameter

        Returns
        ----------
        the cosmo chi2

        """
        Om, w0, M, alpha, beta = tup

        return np.sum((self.Mb-self.mu(self.Z, Om, w0, wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+gzero**2+self.sigMu(Om, w0, wa)**2))

    def zchii2_nowa_im(self, Om, w0, M, alpha, beta):
        """
        Method to estimate the chi2 for a cosmo fit

        Parameters
        --------------
        tup: tuple
          parameters to fit (Om, w0, M, alpha, beta)
         wa: float
           wa parameter
         gzero: float
           gzero parameter

        Returns
        ----------
        the cosmo chi2

        """
        #Om, w0, M, alpha, beta = tup
        return np.sum((self.Mb-self.mu(self.Z, Om, w0, self.wa)-M+alpha*self.X1-beta*self.Color)**2/(self.sigmI(alpha, beta)+self.gzero**2+self.sigMu(Om, w0, self.wa)**2))

    def zfinal1(self, gzero):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0, wa
        """
        return optimize.minimize(self.zchii2, (0.3, -1.0, 0., -19., 0.13, 3.1), args=(gzero), method='BFGS')

    def zfinal1_nowa(self):
        """
        Method to estimate cosmo fit parameters M, alpha, beta, Om, w0

        Parameters
        --------------
        wa: float
          wa DE parameter
        gzero: float
           gzero parameter

        Returns
        ----------
        fitted parameters

        """

        m = Minuit(self.zchii2_nowa_im, Om=0.3, w0=-1.0, M=-19.,
                   alpha=0.13, beta=3.1)
        m.migrad()
        values = m.values
        print('after fit', values)
        m.hesse()   # run covariance estimator
        print(m.errors)
        return values
        # return optimize.minimize(self.zchii2_nowa, (0.3, -1.0, -19., 0.13, 3.1), args=(wa, gzero), method='BFGS')

    def zchi2ndf(self, gzero):
        """
        Method to estimate the gzero parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        gzero: float
          gzero param

        Returns
        ----------
        fitted parameters
        """
        om, w0, wa, m, xi1, xi2 = self.zfinal1(gzero)
        return self.zchii2((om, w0, wa, m, xi1, xi2), gzero)-self.ndf

    def zfinal2(self):
        """
        Method to estimate the gzero parameter
        to get a chisquare equal to 1

        Returns
        ----------
        gzero parameter
         """
        return optimize.newton(self.zchi2ndf, 0.1)

    def zchi2ndf_nowa(self, gzero, wa):
        """
        Method to estimate the gzero parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        wa: float
          wa cosmo parameter
        gzero: float
          gzero param

        Returns
        ----------
        fitted parameters
        """
        res = self.zfinal1_nowa(gzero, wa)
        om, w0,  m, xi1, xi2 = res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]
        #om, w0,  m, xi1, xi2 = 0.3, -1.0, -19.045, 0.13, 3.1
        print('jjj', om, w0,  m, xi1, xi2, gzero, self.ndf)
        return self.zchii2_nowa((om, w0, m, xi1, xi2), wa, gzero)-self.ndf

    def zfinal2_nowa(self, wa):
        """
        Method to estimate the gzero parameter
        to get a chisquare equal to 1

        Parameters
        ---------------
        wa : float
          wa cosmo parameter

        Returns
        ----------
        gzero parameter
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

    def plot_hubble(self, gzero, Om, w0, wa, M, alpha, beta):
        """
        Method to perform a Hubble plot

        Parameters
        --------------
        gzero: float
          gzero parameter
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
                     yerr=np.sqrt(self.sigmI(alpha, beta)+gzero**2+self.sigMu(Om, w0, wa)**2), xerr=None, fmt='o')
        z = np.arange(0.001, 1., 0.001)
        r = self.mu(z, Om, w0, wa)
        plt.plot(z, r+M, 'x')
        plt.show()


class Sigma_Fisher(CosmoDist):
    """"
    class to estimate error parameters using Fisher matrices

    """

    def __init__(self, data,
                 params=dict(zip(['M', 'alpha', 'beta', 'Om', 'w0',
                             'wa'], [-19.045, 0.13, 2.96, 0.3, -1.0, 0.0])),
                 params_Fisher=['M', 'alpha', 'beta', 'Om', 'w0'],
                 H0=72, c=299792.458):
        super().__init__(H0, c)

        Z = data['z_fit']
        Mb = data['mbfit']
        Cov_mbmb = data['Cov_mbmb']
        Cov_x1x1 = data['Cov_x1x1']
        Cov_colorcolor = data['Cov_colorcolor']
        Cov_x1mb = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        Cov_x1mb = data['Cov_x1mb']
        Cov_colormb = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        Cov_colormb = data['Cov_colormb']
        Cov_x1color = data['Cov_x1color']
        X1 = data['x1_fit']
        Color = data['color_fit']
        sigZ = 0.01*(1.+Z)
        #sigZ = 0.0
        # instance of the fit functions here
        self.fit = FitCosmo(Z, X1, Color, Mb, Cov_mbmb,
                            Cov_x1x1, Cov_colorcolor, Cov_x1mb,
                            Cov_colormb, Cov_x1color, sigZ, len(data),
                            params_fit=['M', 'alpha', 'beta', 'Om', 'w0'])
        # fit parameters
        self.params = params
        self.params_Fisher = params_Fisher

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
        # get Fisher matrix
        A = self.matFisher()
        detmat = np.linalg.det(A)
        res = None
        # if detmat > 0:
        # invert this matrix
        #b = np.identity(A.shape[1], dtype=A.dtype)
        # u, piv, x, info = lapack.dgesv(A, b)
        if (np.linalg.det(A)):
            res = np.linalg.inv(A)
            #print('Fisher inverted',res)
            print('Cov from Fisher', np.sqrt(np.diag(res)))
            """
            # marginalize over M, alpha and beta
            mat = np.zeros((2,2))
            mat[0,0] = res[3,3]
            mat[1,0] = res[3,4]
            mat[0,1] = res[4,3]
            mat[1,1] = res[4,4]
            print('after marginalization',mat, np.linalg.inv(mat))
            """

    def matFisher(self):

        parName = self.params_Fisher
        #parName = ['Om','w0','wa']
        nparams = len(parName)

        # get derivatives
        #deriv = self.derivative_chi2(parName)
        # fill the Fisher matrix
        Fisher_Matrix = np.zeros((nparams, nparams))

        for ia, vala in enumerate(parName):
            for jb, valb in enumerate(parName):
                if jb >= ia:
                    deriv = 0
                    derivn = 0
                    if vala == valb:
                        deriv = self.derivative2_same(
                            self.chi2, self.params, vala)
                    else:
                        deriv = self.derivative2_mix(
                            self.chi2, self.params, vala, valb)

                    # print('hello',ia,jb,vala,valb,deriv,derivn)
                    Fisher_Matrix[ia, jb] = 0.5*deriv

        Fisher_Matrix = Fisher_Matrix + np.triu(Fisher_Matrix, 1).T
        """
        for ia, vala in enumerate(parName):
            for jb, valb in enumerate(parName):
                Fisher_Matrix [ia, jb] = 0.5*deriv[vala]*deriv[valb]
        """

        #Fisher_Matrix = Fisher_Matrix + np.triu(Fisher_Matrix, 1).T

        return Fisher_Matrix

    def derivative(self, func, params, parName, h=1.e-8):

        deriv = (func(params, parName, h)-func(params, parName, h))/(2*h)

        return deriv

    def derivative2_same(self, func, params, parName, h=1.e-8):

        deriv = (func(params, parName, h)-2.*func(params) +
                 func(params, parName, -h))/(h**2)

        return deriv

    def derivative2_mix(self, func, params, parName1, parName2, h=1.e-8):

        deriv = func(params, parName1, h, parName2, h)
        deriv -= func(params, parName1, -h, parName2, h)
        deriv -= func(params, parName1, h, parName2, -h)
        deriv += func(params, parName1, -h, parName2, -h)
        deriv /= 4*h*h

        return deriv

    def chi2(self, param, parNamea='', ha=0, parNameb='', hb=0):

        pars = {}
        for key, pp in param.items():
            pars[key] = pp + ha*(parNamea == key)+hb*(parNameb == key)

        tup = (pars['Om'], pars['w0'], pars['M'], pars['alpha'], pars['beta'])
        wa = pars['wa']
        #print('in chi2',M, alpha, beta, Om, w0, wa,np.sum(self.sigmasq(alpha,beta)))
        # return np.sum((self.Mb-M+alpha*self.X1-beta*self.X2-self.mu(self.Z, Om, w0, wa))**2/(self.sigmasq(alpha,beta)+self.sigMu(self.Z,Om,w0,wa,self.sigZ)**2))
        return self.fit.zchii2_nowa(tup, wa, 0.0)

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
