import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import glob
from sn_tools.sn_io import loopStack_params
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
import pandas as pd
from scipy.interpolate import interp1d
from sn_tools.sn_calcFast import faster_inverse


def loadData(dirFile, dbName, tagprod):

    search_path = '{}/{}/*{}*.hdf5'.format(dirFile, dbName, tagprod)
    print('search path', search_path)
    fis = glob.glob(search_path)

    print(fis)
    # load the files
    params = dict(zip(['objtype'], ['astropyTable']))

    return multiproc(fis, params, loopStack_params, 4).to_pandas()


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
            lambda x: self.zcomp095(x))

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

        idxb = grp['fitstatus'] == 'fitok'
        idxb &= np.sqrt(grp['Cov_colorcolor']) <= 0.04
        selb = grp[idxb].to_records(index=False)
        selb.sort(order=['z'])

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

    def __init__(self, H0=72, c=299792.458):

        self.H0 = H0
        self.c = c

    def func(self, z, Om=0.3, w0=-1.0, wa=0.0):
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
        norm = self.c*(1.+z)/self.H0

        return norm*integrate.quad(lambda x: self.func(x, Om, w0, wa), 0.0001, z)[0]

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

        return 5.*np.log10(self.dL(z, Om, w0, wa))+25.

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

        data = self.select(data)
        idx = data['z']-data['zcomp'] <= 0.
        idx = np.abs(data['z']-data['zcomp']) >= 0.

        self.data = data[idx].copy()
        print('Number of SN', len(self.data))
        self.NSN = int(len(self.data)/rescale_factor)

    def select(self, dd):
        """"
        Method to select data

        Parameters
        ---------------
        dd: pandas df
          data to select

        Returns
        -----------
        selected pandas df

        """
        idx = dd['z'] < 1.
        idx &= dd['z'] >= 0.01
        idx &= dd['fitstatus'] == 'fitok'
        idx &= np.sqrt(dd['Cov_colorcolor']) < 0.04

        return dd[idx].copy()

    def plot_sn_vars(self):
        """
        plot data related to supernovae: x1, color and diff with fits

        """

        fig, ax = plt.subplots(ncols=2, nrows=2)

        vars = ['x1', 'x1_fit', 'color', 'color_fit']

        ax[0, 0].hist(self.data_all['x1'], bins=100, histtype='step')
        ax[0, 1].hist(self.data_all['color'], bins=100, histtype='step')

        idx = self.data_all['fitstatus'] == 'fitok'
        sel = self.data_all[idx]
        ax[1, 0].hist(sel['x1']-sel['x1_fit'],
                      bins=20, histtype='step')
        ax[1, 1].hist(sel['color'] -
                      sel['color_fit'], bins=20, histtype='step')

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
        for z in np.arange(zmin, zmax+0.01, 0.01):
            r.append((z, self.mu(z)))

        res = np.rec.fromrecords(r, names=['z', 'mu'])

        # fix, ax = plt.subplots()
        fig = plt.figure()
        # figb, axb = plt.subplots()
        ax = fig.add_axes((.1, .3, .8, .6))

        ax.plot(res['z'], res['mu'], color='b')

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
            x, y, yerr = self.binned_data(zmin, zmax, data, nbins)
            mu_th_binned = res_interp(x)
            residuals = (y-mu_th_binned)
            io = x >= 0.3
            io &= x <= 0.4
            print('mean residual', np.mean(residuals[io]))
        else:
            pp = data.to_records(index=False)
            x, y, yerr = pp['z'], pp['mu'], pp['sigma_mu']
            residuals = pp['mu_residual']
        ax.errorbar(x, y, yerr=yerr,
                    color='k', lineStyle='None', marker='o', ms=2)
        ax.grid()

        axb = fig.add_axes((.1, .1, .8, .2))
        axb.errorbar(x, residuals, yerr=yerr, color='k',
                     lineStyle='None', marker='o', ms=2)
        axb.grid()
        plt.show()

    def binned_data(self, zmin, zmax, data, nbins, vary='mu', erry='sigma_mu'):
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
        group = data.groupby(pd.cut(data.z, bins))
        plot_centers = (bins[:-1] + bins[1:])/2
        plot_values = group.mu.mean()
        plot_values = group.apply(lambda x: np.sum(
            x[vary]/x[erry]**2)/np.sum(1./x[erry]**2))
        print(plot_values)
        error_values = group.apply(
            lambda x: 1./np.sqrt(np.sum(1./x[erry]**2)))
        print('error', error_values)

        return plot_centers, plot_values, error_values


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


parser = OptionParser(
    description='Estimate zlim from simulation+fit data')
parser.add_option("--fileDir", type="str",
                  default='/sps/lsst/users/gris/DD/Fit',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName


tt = zcomp_pixels(fileDir, dbName, 'faintSN')

zcomp = tt()

print(zcomp)


fom = FoM(fileDir, dbName, 'allSN', zcomp)
fom.plot_sn_vars()
# plt.show()

Om = 0.3
w0 = -1.0
wa = 0.0
alpha = 0.14
beta = 3.1
Mb = -19.0481
#Mb = -19.07


h = 1.e-8
varFish = ['dOm', 'dw0', 'dalpha', 'dbeta', 'dMb']
parNames = ['Om', 'w0', 'wa', 'alpha', 'beta', 'Mb']
parameters = dict(zip(parNames, [Om, w0, wa, alpha, beta, Mb]))
epsilon = dict(zip(parNames, [0.]*len(parNames)))
for i in range(1):
    data = fom.data.sample(n=fom.NSN)
    fom.plot_data_cosmo(data, alpha=alpha, beta=beta,
                        Mb=Mb, binned=True, nbins=20)

    Fisher = np.zeros((len(varFish), len(varFish)))
    for vv in parNames:
        epsilon[vv] = h
        data['d{}'.format(vv)] = data.apply(
            lambda x: deriv(x, fom, parameters, epsilon)/2/h, axis=1, result_type='expand')
        epsilon[vv] = 0.0

    for i, vva in enumerate(varFish):
        for j, vvb in enumerate(varFish):
            Fisher[i, j] = np.sum((data[vva]*data[vvb])/data['sigma_mu']**2)
    print(Fisher)
    res = np.sqrt(np.diag(faster_inverse(Fisher)))
    print(res)
    print(len(data), 1./(res[0]*res[1]))
