from sn_maf.sn_tools.sn_cadence_tools import TemplateData, TemplateData_x1color
import h5py
from astropy.table import Table
from iminuit import Minuit, describe
import numpy as np
import time
import glob
import matplotlib.pyplot as plt


class least_square:
    def __init__(self, refdata, band, fluxes_obs, fluxes_obs_err, mjd_obs, m5_obs, exptime_obs):
        self.refdata = refdata
        self.band = band
        self.fluxes_obs = fluxes_obs
        self.fluxes_obs_err = fluxes_obs_err
        self.mjd_obs = mjd_obs
        self.m5_obs = m5_obs
        self.exptime_obs = exptime_obs

    def flux(self, param):
        time_ref = time.time()
        fluxes = self.refdata.Fluxes(self.mjd_obs, param)
        # print('hhh', flux/self.fluxes_obs, self.fluxes_obs_err)
        # print(time.time()-time_ref)
        return fluxes

    def simulation(self, param):
        time_ref = time.time()
        tabres = self.refdata.Simulation(
            self.mjd_obs, self.m5_obs, self.exptime_obs, param)
        # print('hhh', flux/self.fluxes_obs, self.fluxes_obs_err)
        print(time.time()-time_ref)
        return tabres

    """
    def __call__(self, z, daymax):
        r = []
        r.append((z, daymax, -100., 100.))
        tab = np.rec.fromrecords(
            r, names=['z', 'DayMax', 'min_rf_phase', 'max_rf_phase'])
        return np.sum((self.fluxes_obs-self.flux(tab))**2/self.fluxes_obs_err**2)
    """

    def __call__(self, z, daymax):
        r = []
        r.append((z, daymax, -100., 100.))
        tab = np.rec.fromrecords(
            r, names=['z', 'DayMax', 'min_rf_phase', 'max_rf_phase', 'x1', 'color'])
        return np.sum((self.fluxes_obs-self.flux(tab))**2/self.fluxes_obs_err**2)


def Simulation(data_obs):

    bands = 'i'
    func = {}
    r = []
    z = data_obs.meta['z']
    daymax = data_obs.meta['DayMax']
    r.append((z, daymax, -20., 40.))

    """
    r = []
    mjd_min = data_obs['time'].min()
    mjd_max = data_obs['time'].max()
    z_vals = np.arange(0.01, 1., 0.05)
    daymax_vals = np.arange(mjd_min, mjd_max, 1.)
    for z in z_vals:
    for daymax in daymax_vals:
    r.append((z, daymax, -20., 40.))
    """
    """
    r.append((z+0.1, daymax, -20., 40.))
    r.append((z, daymax+1, -20., 40.))
    r.append((z+0.1, daymax+1, -20., 40.))
    """
    time_ref = time.time()
    param = np.rec.fromrecords(
        r, names=['z', 'DayMax', 'min_rf_phase', 'max_rf_phase'])
    for band in bands:
        idx = data_obs['band'] == 'LSST::'+band
        obs = data_obs[idx]
        templdata = TemplateData('LC_-2.0_0.2.hdf5', band)
        # templdata = TemplateData('LC_Test_today.hdf5', band)
        func[band] = least_square(templdata, band,
                                  obs['flux'], obs['fluxerr'], obs['time'], obs['m5'], obs['exptime'])
        tab = func[band].simulation(param)
        print(band, len(tab))
        # print(len(z_vals)*len(daymax_vals))
        print(band, obs['flux'], obs['fluxerr'])
        print((obs['flux']-tab['flux'])/obs['flux'],
              (obs['fluxerr']-tab['fluxerr'])/obs['fluxerr'])

    print('end of simulation', time.time()-time_ref)


def Fit(data_obs, x1, color):
    thedir = 'templates'

    list_files = glob.glob(thedir+'/LC_'+str(x1)+'_'+str(color)+'.hdf5')
    """
    list_file = ['LC_-2.0_0.2.hdf5', 'LC_0.0_0.0.hdf5', 'LC_2.0_-0.2.hdf5',
                 'LC_-2.0_-0.2.hdf5', 'LC_2.0_0.2.hdf5', 'LC_-2.0_0.0.hdf5', 'LC_2.0_0.0.hdf5']
    list_files = [thedir+'/'+ll for ll in list_file]
    """
    print(list_files)
    bands = 'rizy'
    func = {}
    bands_data = []
    for band in bands:
        idx = data_obs['band'] == 'LSST::'+band
        obs = data_obs[idx]
        print('obs', obs['phase'], len(obs))
        plt.plot(obs['time'], obs['flux'])
        plt.show()
        if len(obs) > 0:
            # Load the template data
            # templdata = TemplateData('LC_Test_today.hdf5', band)
            templdata = TemplateData(list_files[0], band)
            # templdata = TemplateData_x1color(list_files, band)
            # print(templdata.refdata)
            func[band] = least_square(templdata, band,
                                      obs['flux'], obs['fluxerr'], obs['time'], obs['m5'], obs['exptime'])
            bands_data.append(band)

    print(bands_data)

    def f(z, daymax):
        funct = np.sum([func[band](z, daymax) for band in bands_data])
        # return func['r'](z, daymax)+func['i'](z, daymax)
        return funct

    z_init = 0.01
    daymax_init = 59968
    x1_init = 0.0
    color_init = 0.0

    time_ref = time.time()
    m = Minuit(f, z=z_init, daymax=daymax_init, error_z=0.02,
               error_daymax=1.)
    # limit_z=(z_init-0.05, z_init+0.05), limit_daymax=(daymax_init-10, daymax_init+10), pedantic=False, fix_z=True)

    # m = Minuit(f, z=z_init, daymax=daymax_init)
    m.migrad()
    values = m.values
    # m.hesse()   # run covariance estimator
    # print(m.errors)
    # get the final chi2
    ndof = len(data_obs)-len(values)
    chi2 = f(values['z'], values['daymax'])/ndof
    print('Timing', ndof, time.time()-time_ref)
    return values, chi2


# Load LC points
fname = 'Output_Simu/LC_DD_baseline2018a_Cosmo.hdf5'
f = h5py.File(fname, 'r')

keys = list(f.keys())

print(keys)

data_obs = Table.read(fname, path=keys[0])

print(data_obs['band'], data_obs.meta)
# Simulation(data_obs)
x1 = -2.0
color = 0.2

list_files = glob.glob('templates/LC*')
zref = data_obs.meta['z']
x1ref = data_obs.meta['X1']
colorref = data_obs.meta['Color']
daymaxref = data_obs.meta['DayMax']
r = []
list_files = ['LC_0.0_0.0.hdf5', 'LC_-2.0_0.0.hdf5', 'LC_-2.0_0.2.hdf5', 'LC_-2.0_-0.2.hdf5',
              'LC_2.0_0.0.hdf5', 'LC_2.0_0.2.hdf5', 'LC_2.0_-0.2.hdf5', 'LC_0.0_0.2.hdf5', 'LC_0.0_-0.2.hdf5']

list_files = ['LC_-2.0_0.2.hdf5']
process = True
if process:
    for fi in list_files:
        #strpl = fi.split('/')[1].split('_')
        strpl = fi.split('_')
        x1 = float(strpl[1])
        color = float(strpl[2].split('.hdf5')[0])
        print('processing', fi, x1, color)
        values, chi2 = Fit(data_obs, x1, color)
        r.append((zref, x1ref, colorref, daymaxref,
                  values['z'], values['daymax'], chi2, x1, color))
        # break
    res = np.rec.fromrecords(r, names=['z_simu', 'x1_simu', 'color_simu',
                                       'daymax_simu', 'z_fit', 'daymax_fit', 'chi2', 'x1_templ', 'color_templ'])

    np.save('test_fit.npy', res)

res = np.load('test_fit.npy')

plt.hist(res['chi2'])
print(res[['z_fit', 'daymax_fit', 'chi2']])
plt.show()
