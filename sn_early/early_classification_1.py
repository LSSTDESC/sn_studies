from sn_maf.sn_tools.sn_cadence_tools import TemplateData
import h5py
from astropy.table import Table
from iminuit import Minuit, describe
import numpy as np


class least_square:
    def __init__(self, refdata, band, fluxes_obs, fluxes_obs_err, mjd_obs):
        self.refdata = refdata
        self.band = band
        self.fluxes_obs = fluxes_obs
        self.fluxes_obs_err = fluxes_obs_err
        self.mjd_obs = mjd_obs

    def func(self, z, daymax):

        flux, fluxerr = self.refdata.EstimateValues(
            self.band, self.mjd_obs, np.array([z]), np.array([daymax]))
        #print('hhh', flux/self.fluxes_obs, self.fluxes_obs_err)
        return flux

    def __call__(self, z, daymax):
        return np.sum((self.fluxes_obs-self.func(z, daymax))**2/self.fluxes_obs_err**2)

# Load the template data


templdata = TemplateData('LC_Test_today.hdf5')

print(templdata.refdata)

# Load LC points

fname = 'LC_DD_baseline2018a_Cosmo_unique_last.hdf5'
f = h5py.File(fname, 'r')

keys = list(f.keys())

print(keys)

data_obs = Table.read(fname, path=keys[0])

print(data_obs['band'], data_obs.meta, data_obs.dtype)

"""
bands = 'gri'
func = {}
for band in bands:
    idx = data_obs['band'] == 'LSST::'+band
    obs = data_obs[idx]
    func[band] = least_square(templdata, band,
                              obs['flux'], obs['fluxerr'], obs['time'])


def f(z, daymax):
    funct = np.sum([func[band](z, daymax) for band in bands])
    # return func['r'](z, daymax)+func['i'](z, daymax)
    return funct


z_init = 0.01
daymax_init = 59968

m = Minuit(f, z=z_init, daymax=daymax_init, error_z=0.02,
           error_daymax=1., limit_z=(z_init-0.05, z_init+0.05), limit_daymax=(daymax_init-10, daymax_init+10), pedantic=False, fix_z=True)
m.migrad()
print(m.values)
m.hesse()   # run covariance estimator
print(m.errors)
"""
