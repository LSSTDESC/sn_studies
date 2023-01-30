import numpy as np
import os
import pandas as pd
from sn_tools.sn_io import loopStack
from sn_telmodel.sn_telescope import Telescope


class VisuLC:
    """"
    class to plot LC from templates

    Parameters
    --------------
    x1: float, opt
     SN x1 (default: -2.0)
    color: float, opt
     SN color (default: 0.2)
    dirTemplates: str, opt
      location dir of the template files (default: Templates)
    cadence: int, opt
      cadence for the templates (default: 1 day)
    error_model: int, opt
      error model value (default: 1)
    error_model_cut: float, opt
      errodel err relative cut (default: 5%)
    bluecutoff: float, opt
      blue cutoff value (default: 380.)
    redcutoff: float, opt
      red cutoff value (default: 800.)
    ebvofMW: float, opt
      E(B-V) (default: 0.0)
    sn_simulator: str, opt
      simulator used to generate the templates (default: sn_fast)

    """

    def __init__(self, x1=-2., color=0.2,
                 dirTemplates='Templates',
                 cadence=1,
                 error_model=1,
                 error_model_cut=0.05,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast'):

        # telescope
        self.telescope = Telescope(airmass=1.2)
        self.x1 = x1
        self.color = color

        cutoff = self.cut_off(error_model, bluecutoff, redcutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0_{}.hdf5'.format(
            sn_simulator, self.x1, self.color, ebvofMW, cutoff, int(cadence), np.round(error_model_cut, 2))

        # load lc
        self.lc = self.load_data(dirTemplates, lcName)

    def load_data(self, theDir, fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str
          name of the input LC file

        Returns
        -----------
        pandas df with LC point infos (flux, fluxerr, ...)
        corresponding to (x1,color) parameters
        """

        # get LC

        lcData = self.load(theDir, fname)

        # select data for the type of SN (x1,color)

        idx = (lcData['x1']-self.x1) < 1.e-1
        idx &= (lcData['color']-self.color) < 1.e-1
        lcData = lcData[idx]

        # remove lc points with negative flux

        idx = lcData['flux_e_sec'] >= 0
        lcData = lcData[idx]

        # transform initial lcData to a pandas df

        lcdf = pd.DataFrame(np.copy(lcData))
        lcdf['band'] = lcdf['band'].map(lambda x: x.decode()[-1])

        return lcdf

    def load(self, theDir, fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str
          name of the input LC file

        Returns
        -----------
        astropy table with LC point infos (flux, fluxerr, ...)
        """

        searchname = '{}/{}'.format(theDir, fname)
        name, ext = os.path.splitext(searchname)

        print(searchname)
        res = loopStack([searchname], objtype='astropyTable')

        return res

    def cut_off(self, error_model, bluecutoff, redcutoff):
        """
        Method to assess error_model or cutoff as str

        Parameters
        ---------------
        error_model: int
          error model value
        bluecutoff: float
          blue cutoff value
        redcutoff: float
          red cutoff value

        Returns
        ----------
        a string with infos (error_model ot bluecutoff_redcutoff)

        """

        cuto = '{}_{}'.format(bluecutoff, redcutoff)
        if error_model:
            cuto = 'error_model'

        return cuto

    def plot(self, z):
        """
        method to plot lc curves corresponding to a given redshift

        Parameters
        ---------------
        z: float
          redshift fot the plot

        """
        import matplotlib.pyplot as plt

        self.lc['snr_photo'] = self.lc['flux']/self.lc['fluxerr_photo']
        idx = np.abs(self.lc['z']-z) < 1.e-5
        idx &= self.lc['snr_photo'] >= 1
        sel_lc = self.lc[idx]

        fig, ax = plt.subplots(ncols=2, nrows=3)

        pos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        bands = 'ugrizy'
        ipos = dict(zip(bands, pos))

        print(sel_lc.columns, sel_lc['z'].unique())
        for b in bands:
            idf = sel_lc['band'] == b
            sel = sel_lc[idf]
            ix = ipos[b][0]
            iy = ipos[b][1]
            print(b, ix, iy, len(sel))
            if len(sel) >= 1:
                ax[ix, iy].errorbar(
                    sel['time'], sel['flux_e_sec'], yerr=sel['flux_e_sec']/sel['snr_photo'])

        plt.show()

    def plotBand(self, z, band, Nvisits=1, m5_singleExposure=26.5):
        """
        method to plot LC curves for redshift, band, nvisits, m5singleExposure

        Parameters
        ---------------
        z: float
          redshift
        band: str
           filter
        Nvisits: int
          number of visits
        m5_singleExposure: float
          5-sigma depth single exposure

        """
        import matplotlib.pyplot as plt

        self.lc['snr_photo'] = self.lc['flux']/self.lc['fluxerr_photo']
        idx = np.abs(self.lc['z']-z) < 1.e-5
        idx &= self.lc['snr_photo'] >= 1
        idx &= self.lc['band'] == band
        sel_lc = self.lc[idx]
        print('SNRa', np.sqrt(np.sum(sel_lc['snr_photo']**2)))
        # need to reestimate flux error
        m5 = m5_singleExposure+1.25*np.log(Nvisits)
        print('there I am', m5, m5_singleExposure, sel_lc['m5'].unique())
        f5 = self.telescope.mag_to_flux(m5, band)
        sel_lc['fluxerr_photo'] = f5/5.
        sel_lc['snr_photo'] = sel_lc['flux_e_sec']/sel_lc['fluxerr_photo']
        idx &= sel_lc['snr_photo'] >= 1
        sel = sel_lc[idx]
        print('SNRb', np.sqrt(np.sum(sel['snr_photo']**2)))

        fig, ax = plt.subplots()

        ax.errorbar(
            sel['time'], sel['flux_e_sec'], yerr=sel['flux_e_sec']/sel['snr_photo'])

        plt.show()
