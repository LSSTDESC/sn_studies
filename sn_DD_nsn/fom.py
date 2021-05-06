import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import glob
from sn_tools.sn_io import loopStack_params
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
import pandas as pd
from scipy.interpolate import interp1d


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
    H0 = 72.  # km.s-1.Mpc-1
    c = 299792.458  # km.s-1
    """

    def __init__(self, H0=72, c=299792.458, Om=0.3, w0=-1., wa=0.):

        self.H0 = H0
        self.c = c
        self.Om = Om
        self.w0 = w0
        self.wa = wa

    def func(self, z):

        wp = self.w0+self.wa*z/(1.+z)

        H = self.Om*(1+z)**3+(1.-self.Om)*(1+z)**(3*(1.+wp))
        fu = np.sqrt(H)

        return 1/fu

    def dL(self, z):
        """
        Method to estimate the luminosity distance

        Parameters
        ---------------
        z: float
           redshift

        Returns
        ----------
        luminosity distance
        """
        norm = self.c*(1.+z)/self.H0

        return norm*integrate.quad(lambda x: self.func(x), 0.01, z)[0]

    def mu(self, z):
        """
        Method to estimate distance modulus

        Parameters
        ---------------
        z: float 
           redshift

        Returns
        -----------
        distance modulus (float)

        """
        return 5.*np.log10(self.dL(z))+25.


class FoM(CosmoDist):
    """


    """

    def __init__(self, fDir, dbName, tagprod, H0=72, c=299792.458, Om=0.3, w0=-1., wa=0.):
        super().__init__(H0, c, Om, w0, wa)

        # load data
        self.data = loadData(fDir, dbName, tagprod)

    def plot_data_cosmo(self):

        r = []
        for z in np.arange(0.01, 1., 0.01):
            r.append((z, self.mu(z)))

        res = np.rec.fromrecords(r, names=['z', 'mu'])

        plt.plot(res['z'], res['mu'], color='b')
        idx = self.data['z'] < 1.0
        idx &= self.data['fitstatus'] == 'fitok'
        idx &= np.sqrt(self.data['Cov_colorcolor']) < 0.04
        sel = self.data[idx].copy()
        print(sel.columns)
        sel['mu'] = sel['mbfit']+0.13 * \
            sel['x1_fit']-3.*sel['color_fit']+19.
        if 'mu' in sel.columns:
            bins = np.linspace(0, 1, 100)
            group = sel.groupby(pd.cut(sel.z, bins))
            plot_centers = (bins[:-1] + bins[1:])/2
            plot_values = group.mu.mean()
            plt.plot(plot_centers, plot_values, 'ko')
            """
            pp = sel.to_records(index=False)
            plt.plot(pp['z'], pp['mu'], 'ko')
            plt.hist2d(pp['z'], pp['mu'], bins=20)
            """
        plt.show()


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

"""
tt = zcomp_pixels(fileDir, dbName, 'faintSN')

zcomp = tt()

print(zcomp)
"""

fom = FoM(fileDir, dbName, 'allSN')

fom.plot_data_cosmo()
