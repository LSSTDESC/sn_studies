import h5py
from astropy.table import Table, vstack
import numpy as np
from sn_tools.sn_io import loopStack
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
from scipy import stats


class ErrorModel_Study:
    """
    class to study errormodel effects

    Parameters
    --------------
    theDir: str
      location dir of the file of interest
    theFile: str
      Simu file to process

    """

    def __init__(self, theDir, theFile):

        self.theDir = theDir
        self.theFile = theFile
        self.lcName = '{}/{}'.format(theDir, theFile.replace('Simu', 'LC'))
        self.bands = 'grizy'

        # flux(e/sec) to mag estimation
        self.flux_e_secTomag = self.flux_to_mag()
        # load m5 obs to estimate the number of visits per band
        m5_obs = pd.DataFrame(np.load(
            'dd_design_test/m5_files/medValues_flexddf_v1.4_10yrs_DD.npy', allow_pickle=True))
        self.m5_single = m5_obs.groupby(
            ['filter'])['fiveSigmaDepth'].median().reset_index()

        # load simu file
        simus = self.loadSimu()

        # loop on simus and plot
        self.plot(simus)

    def loadSimu(self):
        """
        Method to load Simu file

        Returns
        ----------
        astropy table with a list of performed simulations

        """
        fname = '{}/{}'.format(self.theDir, self.theFile)

        print('loading here', fname)
        simus = loopStack([fname], 'astropyTable')

        return simus

    def flux_to_mag(self):
        """
        Method to estimate the mag<->flux conv

        Returns
        -----------
        flux_e_sec_to_mag: dict
          dict of interp1d(flux->mag), keys=bands.
        """
        from sn_telmodel.sn_telescope import Telescope
        telescope = Telescope(airmass=1.2)

        mags = np.arange(15., 30., 0.1)
        bands = 'grizy'
        exptime = 30.
        nexp = 1

        flux_e_sec_to_mag = {}
        for b in bands:
            fluxes_e_sec = telescope.mag_to_flux_e_sec(
                mags, [b]*len(mags), [exptime]*len(mags), [nexp]*len(mags))
            flux_e_sec_to_mag[b] = interpolate.interp1d(
                fluxes_e_sec[:, 1], np.array(mags), fill_value=-1, bounds_error=False)

        return flux_e_sec_to_mag

    def plot(self, simus):
        """
        Method to plot 

        Parameters
        --------------
        simus: astropy table
        data to process

        """
        print('ssss', simus)
        for b in self.bands:
            lc = self.getLCs(b, simus)
            print(lc.columns)
            self.plotBand(lc, whatx='z', whaty='fluxerr_model_rel')

        plt.show()

    def getLCs(self, band, simus):
        """
        Method extracting all LC data corresponding to band

        Parameters
        --------------
        band: str
          band of interest
        simus: astropy table
          simu data

        """

        lctot = Table()
        for val in simus:
            lc = Table.read(
                self.lcName, path='lc_{}'.format(val['index_hdf5']))
            idx = lc['band'] == 'LSST::{}'.format(band)
            lcres = lc[idx]
            lcres['z'] = val['z']
            lcres['SNR_errormodel'] = lcres['flux']/lcres['fluxerr_model']
            lcres['fluxerr_model_rel'] = lcres['fluxerr_model']/lcres['flux']
            lcres['fluxerr_model_e_sec'] = lcres['fluxerr_model'] / \
                lcres['flux']*lcres['flux_e_sec']
            lcres['m5_errmodel'] = self.flux_e_secTomag[band](
                lcres['fluxerr_model_e_sec']/5.)
            idx = self.m5_single['filter'] == band
            lcres['Nvisits_errmodel'] = 10**0.04 * \
                (lcres['m5_errmodel']-self.m5_single[idx]
                 ['fiveSigmaDepth'].item())
            lctot = vstack([lctot, lcres])

        return lctot

    def plotBand(self, tab, whatx, whaty):

        fig, ax = plt.subplots()

        idx = tab['snr_m5'] >= 1.
        idx &= tab['z'] > 0.2
        sel = tab[idx]
        ax.plot(sel[whatx], sel[whaty], 'ko')


def load_Fit(theDir, theFile):
    fName = '{}/{}'.format(theDir, theFile)
    fi = h5py.File(fName, 'r')

    keys = list(fi.keys())

    tab = Table.read(fName, path=keys[0])
    # print(tab.columns)
    # print(tab[['x1_fit','color_fit','z_fit','z','Cov_t0t0']])
    idx = tab['fitstatus'] == 'fitok'
    sel = tab[idx]
    # print(np.mean(tab['x1_fit']),np.std(tab['x1_fit']),np.mean(tab['color_fit']),np.std(tab['color_fit']))
    # print(np.median(tab['x1_fit']),np.median(tab['color_fit']),np.median(tab['mbfit']))
    return tab


def loadLC(theDir, theFile, zref=0.11):

    fname = '{}/{}'.format(theDir, theFile)

    tab = loopStack([fname], 'astropyTable')
    print('looking dor', fname, len(tab))
    lcname = '{}/{}'.format(theDir, theFile.replace('Simu', 'LC'))
    for val in tab:
        lc = Table.read(lcname, path='lc_{}'.format(val['index_hdf5']))
        # print(lc.meta['z'])
        if np.abs(lc.meta['z']-zref) < 1.e-5:
            print(np.max(lc['flux']))
            return lc

    return None


def plotLC(tabs, z, colors, sna, snb):

    pos = dict(zip('ugrizy', [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))

    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.suptitle('z={}'.format(z))

    deltaph = []
    for b in 'grizy':
        deltaph.append(
            (b, deltaphase(sna, snb, z, b, tabs, whatx='phase', whaty='flux')))

    dph = np.rec.fromrecords(deltaph, names=['band', 'deltaphi'])
    print('jjj', dph)
    #dph = np.median(res[res['band']=='r']['deltaphi'])

    for b in 'grizy':
        i = pos[b][0]
        j = pos[b][1]

        for key, tab in tabs.items():
            idx = tab['band'] == 'LSST::'+b
            idx &= tab['snr_m5'] >= 1.
            sel = tab[idx]
            # ax[i,j].plot(sel['time'],sel['flux'],'{}o'.format(colors[io]),mfc=mfc[io])
            deltapph = 0.
            """
            if key == snb:
                deltapph = dph[dph['band']==b]['deltaphi']
            """
            ax[i, j].errorbar(sel['phase']+deltapph, sel['flux'],
                              yerr=sel['fluxerr_model'], color=colors[key], marker='o')
    plt.show()


def interp(tab, whatx='phase', whaty='flux'):

    return interpolate.interp1d(tab[whatx], tab[whaty], bounds_error=False, fill_value=0.)


def deltaphase(sna, snb, z, b, tabs, whatx='phase', whaty='flux', phases=np.arange(-20., 60., 0.1)):

    phasemax = {}
    # print(stats.ttest_ind(tabs[sna]['phase'],tabs[snb]['phase']))
    for key, tab in tabs.items():
        idx = tab['band'] == 'LSST::'+b
        idx &= tab['snr_m5'] >= 1.
        sel = tab[idx]
        flux_interp = interpolate.interp1d(
            sel[whatx], sel[whaty], bounds_error=False, fill_value=0.)
        fluxes = np.asarray(flux_interp(phases))
        #io = np.argwhere(fluxes = np.max(fluxes))
        io = fluxes.argmax()
        phasemax[key] = phases[io]

    return (phasemax[sna]-phasemax[snb])*(1.+z)


theDir = 'Output_Fit_0.0_100000.0_ebvofMW_0.0_snrmin_1'
theFile = 'Fit_sn_cosmo_Fake_SN_IaT_nugent-sn1a_0.0_100000.0_ebvofMW_0.0_sn_cosmo.hdf5'
#theFile = 'Fit_sn_cosmo_Fake_SN_IaT_hsiao_0.0_100000.0_ebvofMW_0.0_sn_cosmo.hdf5'
#theDir = 'Output_Fit_error_model_ebvofMW_0.0_snrmin_1'
#theFile = 'Fit_sn_cosmo_Fake_0.2_-0.1_error_model_ebvofMW_0.0_sn_cosmo.hdf5'

load_Fit(theDir, theFile)

theDir_real = 'Output_Simu_0.0_100000.0_ebvofMW_0.0'
theFile_real = {}
theFile_real['nugent-sn1a'] = 'Simu_sn_cosmo_Fake_SN_IaT_nugent-sn1a_0.0_100000.0_ebvofMW_0.0_0.hdf5'
#theFile_real['hsiao'] = 'Simu_sn_cosmo_Fake_SN_IaT_hsiao_0.0_100000.0_ebvofMW_0.0_0.hdf5'

theDir_simu = 'Output_Simu_error_model_ebvofMW_0.0'
theFile_simu = {}
theFile_simu['salt2_nugent-sn1a'] = 'Simu_sn_cosmo_Fake_0.2_-0.1_error_model_ebvofMW_0.0_0.hdf5'
#theFile_simu['salt2_hsiao'] = 'Simu_sn_cosmo_Fake_0.54_-0.06_error_model_ebvofMW_0.0_0.hdf5'

ErrorModel_Study('Output_Simu_error_model_ebvofMW_0.0',
                 'Simu_sn_cosmo_Fake_0.2_-0.1_error_model_ebvofMW_0.0_0.hdf5')

print(test)
zref = 0.26
lc = {}
for key, vals in theFile_real.items():
    lc[key] = loadLC(theDir_real, vals, zref)

for key, vals in theFile_simu.items():
    lc[key] = loadLC(theDir_simu, vals, zref)

colors_ref = ['b', 'r', 'k', 'k']
colors = dict(zip(lc.keys(), colors_ref[:len(lc.keys())]))

plotLC(lc, zref, colors, 'nugent-sn1a', 'salt2_nugent-sn1a')

# print(tab)
