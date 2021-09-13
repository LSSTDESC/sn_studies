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
                s[i] = (1+t)*quad(integrand, 0, t,limit=100)[0]
            return s
        else:
            return (1+z)*quad(integrand, 0, z,limit=100)[0]

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


class FitCosmo(CosmoDist):
    """
    Class to fit cosmology from a set of data

    Parameters
    ---------------


    """

    def __init__(self, data, H0=72, c=299792.458):
        super().__init__(H0, c)

        print('Number of SN for fit', len(data))
        # print set([d[name]['idr.subset'] for name in d.keys()]

        print(data.columns)
        self.Z = data['z_fit']
        self.Mb = data['mbfit']
        self.Mber = np.sqrt(data['Cov_mbmb'])
        self.gx1 = data['Cov_x1x1']
        self.gxc = data['Cov_colorcolor']
        self.cov1 = -2.5*data['Cov_x0x1'] / \
            (data['x0_fit']*np.log(10))
        self.cov1 = data['Cov_x1mb']
        self.cov2 = -2.5*data['Cov_x0color'] / \
            (data['x0_fit']*np.log(10))
        self.cov2 = data['Cov_colormb']
        self.cov3 = data['Cov_x1color']
        self.sigZ = 1.e-3*data['z_fit']
        self.X1 = data['x1_fit']
        self.X2 = data['color_fit']
        self.ndf = len(data)-5
        """
        data['sigma_mu_recalc'] = np.sqrt(self.sigmI(0.14, 3.1))
        import matplotlib.pyplot as plt
        plt.plot(data['z_fit'], data['sigma_mu'], 'ko')
        plt.plot(data['z_fit'], data['sigma_mu_recalc'], 'ro')
        plt.show()
        """

    def __call__(self):
        """
        Method to perform the cosmological fit


        """

        # initial parameter values
        gzero = 0.273409368886
        gzero = 0.
        ol = 0.84943991855238044
        Om = 0.3
        w0 = -1.0
        wa = 0.0
        m = -19.260197289410925
        xi1 = -0.13514318369819331
        xi2 = 1.8706300018097486

        # first estimate of Mb
        mu = self.mu(self.Z, Om, w0, wa)

        mean = np.sum((self.Mb-mu)/self.Mber**2 / np.sum(1/self.Mber**2))
        print(mean)

        #
        self.sigZ = 5*self.sigZ/(self.Z*np.log(10))

        time_ref = time.time()
       # the fit is done here
        res = self.zfinal1(
            self.Mb, self.Z, self.X1, self.X2, self.sigZ)

        print('after fit', time.time()-time_ref)
        print(res.x)
        # get the covariance matrix
        covmat = res.hess_inv
        #covmat = res.jac
        print(np.sqrt(np.diag(covmat)))

        # Extract the parameters and copy in a dict
        params = {}

        parNames = ['Om', 'w0', 'wa', 'M', 'alpha', 'beta']

        for i, vv in enumerate(parNames):
            params[vv] = [res.x[i]]
            for j, pp in enumerate(parNames):
                if j >= i:
                    nn = 'Cov_{}_{}'.format(vv, pp)
                    params[nn] = [covmat[i][j]]

        tup = (params['Om'][0],params['w0'][0],
               params['wa'][0], params['M'][0],
               params['alpha'][0], params['beta'][0])
        print('hhh',tup)
        chi2 = self.zchii2(tup, self.Mb, self.Z, self.X1, self.X2, self.sigZ)/self.ndf
        params['chi2' ] = [chi2]
        
        fit_result = pd.DataFrame.from_dict(params)

        return fit_result

    def luminosity_distance(self, Z, Ol):
        """ Returns the product of H0 and D_l, the luminosity distance
        Z : range of redshifts
        Ol : Omega_Lambda """
        def integrand(x): return 1/np.sqrt(Ol+(1-Ol)*(1+x)**3)
        if (hasattr(Z, '__iter__')):
            s = np.zeros(len(Z))
            for i, t in enumerate(Z):
                s[i] = (1+t)*quad(integrand, 0, t)[0]
            return s
        else:
            return (1+Z)*quad(integrand, 0, Z)[0]

    """
    def distance_modulus(self, Z, Ol = 0.72):
        if (hasattr(Z, '__iter__')):
            return np.log10(self.luminosity_distance(Z, Ol)*3e11/72)*5-5
        else:
            return (np.log10(self.luminosity_distance([Z], Ol)*3e11/72)*5-5)[0]

    def distance_modulus(self, Z, Om, w0, wa):
        if (hasattr(Z, '__iter__')):
            return np.log10(self.dL(Z, Om, w0, wa))*5-5
        else:
            return (np.log10(self.dL([Z], Om, w0, wa))*5-5)[0]
        """

    def plot_modulus(self, offset=0):
        z = np.arange(0.001, 0.12, 0.001)
        for Ol in [0, 0.3, 0.7, 1]:
            # hodl=luminosity_distance(z,Ol)
            r = self.mu(z, Ol)
            plt.plot(z, hodl)
            plt.plot(z, r+offset, 'x')
        plt.show()

    def sigmI(self, alpha, beta):

        # return self.Mber**2+(Xi1**2)*self.gx1+(Xi2**2)*self.gxc-2*Xi1*self.cov1-2*Xi2*self.cov2+2*Xi1*Xi2*self.cov3
        return self.Mber**2+(alpha**2)*self.gx1+(beta**2)*self.gxc+2*alpha*self.cov1-2*beta*self.cov2-2*alpha*beta*self.cov3

    def sigMu(self, Ol, Z, sigZ):
        return self.derivate2(Z, Ol)*sigZ

    def zchii2(self, tup, Mb, Z, x1, color, sigZ):
        Om, w0, wa, M, alpha, beta = tup
        # return np.sum((Mb-self.mu(Z, Om, w0, wa)-M+alpha*x1-beta*color)**2/(self.sigmI(alpha, beta)+gzero**2+self.sigMu(1.-Om, Z, sigZ)**2))
        return np.sum((Mb-self.mu(Z, Om, w0, wa)-M+alpha*x1-beta*color)**2/(self.sigmI(alpha, beta)))

    def zfinal1(self, Mb, Z, x1, color, sigZ):
        return optimize.minimize(self.zchii2, (0.3, -1.0, 0., -19., 0.13, 3.1), args=(Mb, Z, x1, color, sigZ),method='BFGS')
        #return optimize.least_squares(self.zchii2, (0.3, -1.0, 0., -19., 0.13, 3.1), args=(Mb, Z, x1, color, sigZ))
    def zchi2ndf(self, gzero, Z, Mb, X1, X2, sigZ):
        om, w0, wa, m, xi1, xi2 = self.zfinal1(Mb, Z, gzero, X1, X2, sigZ)
        return self.zchii2((om, w0, wa, m, xi1, xi2), Mb, Z, gzero, X1, X2, sigZ)-119

    def zfinal2(self, Mb, Z, X1, X2, sigZ):
        return optimize.newton(self.zchi2ndf, 0.1, args=(Z, Mb, X1, X2, sigZ))

    def derivate2(self, Z, Ol):
        dl = self.luminosity_distance(Z, Ol)
        integrand = 1/np.sqrt(Ol+(1-Ol)*(1+Z)**3)
        coeff = 5./np.log(10)
        val1 = coeff/(1+Z)
        val2 = coeff*(1+Z)/dl*(integrand)
        return val1+val2

    def plot_hubble(self, gzero, Om, w0, wa, M, Xi1, Xi2):
        plt.errorbar(self.Z, self.Mb+Xi1*self.X1-Xi2*self.X2,
                     yerr=np.sqrt(self.sigmI(Xi1, Xi2)+gzero**2), xerr=None, fmt='o')
        z = np.arange(0.001, 1., 0.001)
        r = self.mu(z, Om, w0, wa)
        plt.plot(z, r+M, 'x')

class Sigma_Fisher(CosmoDist):
    """"
    class to estimate error parameters using Fisher matrices

    """
    def __init__(self, data,
                 params=dict(zip(['M', 'alpha', 'beta', 'Om','w0','wa'],[-19.,0.13,3.1,0.3,-1.0,0.0])), H0=72, c=299792.458):
        super().__init__(H0, c)

        self.Mb = data['mbfit']
        self.x1 = data['x1_fit']
        self.color = data['color_fit']
        self.Z = data['z_fit']
        self.data = data
        # fit parameters
        self.params = params

    def __call__(self):

        # get Fisher matrix
        A = self.matFisher()
        detmat = np.linalg.det(A)
        res = None
        #if detmat > 0:
            # invert this matrix
        #b = np.identity(A.shape[1], dtype=A.dtype)
            # u, piv, x, info = lapack.dgesv(A, b)
        if (np.linalg.det(A)):
            res = np.linalg.inv(A)
        print('Cov from Fisher',res)
         
         
    def matFisher(self):

        parName = list(self.params.keys())
        #parName = ['Om','w0','wa']
        nparams = len(parName)

        # get derivatives
        deriv = self.derivative_chi2(parName)
        
        # fill the Fisher matrix
        Fisher_Matrix = np.zeros((nparams,nparams))
        
        for ia, vala in enumerate(parName):
            for jb, valb in enumerate(parName):
                #if jb >= ia:
                Fisher_Matrix [ia, jb] = 0.5*deriv[vala]*deriv[valb] 
   
        #Fisher_Matrix = Fisher_Matrix + np.triu(Fisher_Matrix, 1).T

        return Fisher_Matrix

    def derivative_chi2(self, parName):
        pp_p = copy.deepcopy(self.params)
        pp_m = copy.deepcopy(self.params)
        deriv = {}
        h = 1.e-8
        for name in parName:
            pp_p[name] += h
            pp_m[name] -=h 
            deriv[name] = (self.chi2(pp_p)-self.chi2(pp_m))/(2*h)
            pp_p[name] -=h
            pp_m[name] +=h
        return deriv
        
    def chi2(self, param):

        M = param['M']
        alpha = param['alpha']
        beta = param['beta']
        Om = param['Om']
        w0 = param['w0']
        wa = param['wa']
        #print('in chi2',M, alpha, beta, Om, w0, wa)
        return np.sum((self.Mb-self.mu(self.Z, Om, w0, wa)-M+alpha*self.x1-beta*self.color)**2/self.sigmasq(alpha,beta))

    def sigmasq(self, alpha, beta):
        
        return self.data['Cov_mbmb']+(alpha**2)*self.data['Cov_x1x1']+(beta**2)*self.data['Cov_colorcolor']+2*alpha*self.data['Cov_x1mb']-2.*beta*self.data['Cov_colormb']-2.*alpha*beta*self.data['Cov_x1color']
