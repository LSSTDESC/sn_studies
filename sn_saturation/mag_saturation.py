from sn_tools.sn_telescope import Telescope
from . import plt
import numpy as np
import numpy.lib.recfunctions as rf
from scipy.interpolate import interp1d


class MagToFlux:
    """
    class used to estimate and plot the mag->flux (in pe/sec)

    Parameters
    --------------
    airmass: float, opt
      arimass value for telescope throughput (default: 1.2)

    """

    def __init__(self, airmass=1.2):

        self.telescope = Telescope(airmass=airmass, aerosol=False)

        # plt.figure()
        # self.telescope.Plot_Throughputs()
        # plt.savefig('LSST_throughput.png')

        mag = np.arange(14, 20.1, 0.1)
        exptime = 15.
        nexp = 1
        self.bands = 'gri'
        self.filtcols = dict(zip(self.bands, 'bgr'))
        self.data = self.loop_band(mag, exptime, nexp)

    def loop_band(self, mag, exptime, nexp):
        """
        Method to estimate the mag-> flux in pe/sec for a set of bands

        Parameters
        --------------
        mag: array(float)
          magnitudes to convert to flux
        exptime: float
          exposure time
        nexp: int
           number of exposures

        Returns
        ----------
        numpy array with the following cols:
        flux_e_sec, band, mag

        """
        restot = None

        for band in self.bands:
            res = self.mag_flux(mag, band, exptime, nexp)
            if restot is None:
                restot = res
            else:
                restot = np.concatenate((restot, res))
        return restot

    def mag_flux(self, mag, band, exptime, nexp):
        """
        Method to estimate the mag-> flux in pe/sec

        Parameters
        --------------
        mag: array(float)
          magnitudes to convert to flux
        band: str
          band considered
        exptime: float
          exposure time
        nexp: int
           number of exposures

        Returns
        ----------
        numpy array with the following cols:
        flux_e_sec, band, mag

        """

        flux_e_sec = self.telescope.mag_to_flux_e_sec(
            mag, [band]*len(mag), [exptime]*len(mag), [nexp]*len(mag))[:, 1]
        # print(band,flux_e_sec)

        res = np.array(flux_e_sec, dtype=[('flux_e_sec', 'f8')])
        res = rf.append_fields(res, 'band', [band]*len(res))
        res = rf.append_fields(res, 'mag', mag)

        return res

    def plot(self, savePlot=False):
        """
        Method to display flux vs mag

        Parameters
        --------------
        savePlot: bool, opt
          to save the plot (default: False)

        """

        #plt.ticklabel_format(style='scientific', axis='y',useMathText=True)
        plt.figure()
        plt.gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))
        for band in self.bands:
            idx = self.data['band'] == band
            sel = self.data[idx]
            plt.semilogy(sel['mag'], sel['flux_e_sec'],
                         color=self.filtcols[band], label='{} band'.format(band))

            #plt.ticklabel_format(style='scientific', axis='y',useMathText=True)
        plt.xlabel('mag')
        plt.ylabel('flux [pe/s]')
        plt.xlim([14., 20.])
        plt.ylim([1e3, 1e6])
        plt.legend()
        plt.grid()
        if savePlot:
            plt.savefig('flux_mag.png')


class MagSaturation:
    """
    class used to estimate the magnitude of saturation vs seeing, fullwell

    Parameters
    --------------
    airmass: float,opt
      airmass for telescope throughput (default: 1.2)
    psf_type: str, opt
      type of PSF used to estimate the pixel max frac (default: single_gauss)
    bands: str, opt
      bands to consider (default: gri)
    """

    def __init__(self, airmass=1.2, psf_type='single_gauss', bands='gri'):

        self.bands = bands
        self.psf_type = psf_type
        # telescope
        self.telescope = Telescope(airmass=airmass, aerosol=False)

        # mag range
        self.mag = np.arange(10, 20, 0.01)

        # seeing range
        self.seeing = np.arange(0.3, 2.49, 0.01)

        # pixel frac max interpolator
        pixel_npy = np.load('PSF_pixel_{}_summary.npy'.format(psf_type))
        pixel_max = interp1d(
            pixel_npy['seeing'], pixel_npy['pixel_frac_med'], fill_value=0.0, bounds_error=False)
        self.pixel_seeing = pixel_max(self.seeing)[:]

    def __call__(self, exptime, full_well):
        """
        method to estimate the saturation magnitude vs exptime and full_well

        Parameters
        ---------------
        exptime: float
          exposure time
        full_well: float
          ccd full well (in pe)

        Returns
        ----------
        numpy array with the following cols:
        seeing, mag, exptime, full_well,band,psf_profile

        """

        res = None
        for b in self.bands:
            print('band', b)
            resb = self.mag_sat_band(b, exptime, full_well)
            if res is None:
                res = resb
            else:
                res = np.concatenate((res, resb))

        return res

    def mag_sat_band(self, band, exptime, full_well):
        """
        Method to estimate mag sat vs band, exptime, full_well

        Parameters
        --------------
        band: str
          band to consider
        exptime: float
          exposure time (sec)
        full_well: float
          ccd full well (in pe)

        Returns
        ----------
        numpy array with the following cols:
        seeing, mag, exptime, full_well,band,psf_profile

        """

        nv = len(self.mag)
        flux_e_sec = self.telescope.mag_to_flux_e_sec(
            self.mag, [band]*nv, [exptime]*nv, [1]*nv)[:, 1]

        all_flux = flux_e_sec[:]*self.pixel_seeing[:, None]*exptime

        flag = all_flux <= full_well
        flag_idx = np.argwhere(flag)

        mag_sat = self.mag[np.argmax(all_flux*flag, axis=1)]

        resu = np.array(self.seeing[:], dtype=[('seeing', 'f8')])
        resu = rf.append_fields(resu, 'mag', mag_sat)
        resu = rf.append_fields(resu, 'exptime', [exptime]*len(resu))
        resu = rf.append_fields(resu, 'full_well', [full_well]*len(resu))
        resu = rf.append_fields(resu, 'band', [band]*len(resu))
        resu = rf.append_fields(resu, 'psf_profile', [self.psf_type]*len(resu))

        return resu


def PlotMagSat(bands, restot, psf_type='single_gauss'):
    """
    Display saturation magnitudes vs seeing (per band)

    Parameters:
    -----------
    bands: bands to consider
    restot: numpy array of results

    Returns:
    --------
    None


    """

    fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(14, 15))
    fig.subplots_adjust(right=0.75)
    ipos = dict(zip(bands, [0, 1, 2]))
    lstyle_all = ['-', ':', '--']
    colors_all = ['b', 'r', 'k']
    exptimes = np.unique(restot['exptime'])
    full_wells = np.unique(restot['full_well'])

    lsstyle = dict(zip(exptimes, lstyle_all[:len(exptimes)]))
    colors = dict(zip(full_wells, colors_all[:len(full_wells)]))
    fontsize = 20
    for band in bands:
        pos = ipos[band]
        for exptime in exptimes:
            for full_well in full_wells:
                sela = select(restot, band, exptime, full_well, psf_type)
                #label = 'exptime: {} s - full well: {} pe'.format(int(exptime),int(full_well))
                label = '{} s / {}k pe'.format(int(exptime),
                                               int(full_well/1000))
                lstyle = lsstyle[exptime]
                col = colors[full_well]
                ax[pos].plot(sela['seeing'], sela['mag'],
                             linestyle=lstyle, color=col, label=label)
                # print(band,exptime,full_well,np.max(sela['mag_sat']),np.max(selb['mag_sat']))
                """
                idx = np.abs(sela['seeing']-0.8).argmin()
                print(band,exptime,full_well,sela[idx]['seeing'],sela[idx]['mag'])
                """
        if pos == 1:
            # ax[pos][0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              # ncol=1, fancybox=True, shadow=True)
            ax[pos].legend(loc='center left', bbox_to_anchor=(
                1, 0.5), ncol=1, fontsize=fontsize)
        ax[pos].set_ylabel('mag', fontsize=fontsize)

        if pos == 2:
            ax[pos].set_xlabel(r'seeing ["]', fontsize=fontsize)
        if pos == 0:
            ax[pos].set_title(r'{}'.format(psf_type), fontsize=fontsize)
            # ax[iw].legend(loc='best',prop={'size':fontsize})
        ax[pos].tick_params(labelsize=fontsize)
        #ax[iw].set_title('full well = {} pe'.format(int(fwell)))
        ax[pos].grid()
        ax[pos].set_xlim([0.3, 2.5])
        ax[pos].xaxis.set_ticks(np.arange(0.3, 2.5, 0.4))
        ax[pos].text(0.8, 0.9, band, horizontalalignment='center',
                     verticalalignment='center', transform=ax[pos].transAxes, fontsize=fontsize)


def select(res, band, exptime, full_well, profile):
    """
    Selection of a numpy array according to some input parameters

    Parameters:
    -----------
    res: input numpy array
    selection criteria:
    - band
    - exptime: exposure time [s]
    - full_well: full well [pe]
    - profile: PSF profile

    Returns:
    -------
    selected array


    """
    idb = res['band'] == band
    idb &= (np.abs(res['exptime']-exptime) < 1.e-5)
    idb &= (np.abs(res['full_well']-full_well) < 1.e-5)
    idb &= res['psf_profile'] == profile
    return res[idb]
