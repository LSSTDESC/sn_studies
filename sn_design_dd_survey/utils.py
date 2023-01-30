from sn_telmodel.sn_telescope import Telescope
import numpy as np
from scipy import interpolate
from sn_tools.sn_io import loopStack
import os
from scipy.interpolate import interp1d


def flux5_to_m5(bands):
    """
    Function to estimate m5 from 5-sigma fluxes

    Parameters
    ----------
    bands: str
     filters considered

    Returns
    -------
    f5_dict: dict
     keys = bands
     values = interp1d(flux_5,m5)
    """

    m5_range = np.arange(15., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    f5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range), [1]*len(m5_range))[:, 1]
        f5_dict[band] = interpolate.interp1d(
            flux_5, m5_range, bounds_error=False, fill_value=0.0)

    return f5_dict


def m5_to_flux5(bands):
    """
    Function to estimate 5-sigma fluxes from m5 values

    Parameters
    ----------
    bands: str
     filters considered

    Returns
    -------
    f5_dict: dict
     keys = bands
     values = interp1d(m5,flux_5)
    """
    m5_range = np.arange(15., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    m5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range),[1]*len(m5_range))[:, 1]
        m5_dict[band] = interpolate.interp1d(
            m5_range, flux_5, bounds_error=False, fill_value=0.0)

    return m5_dict


def srand(gamma, mag, m5):
    """
    Function to estimate sigma_rand = 1./SNR
    see eq (5) of
    LSST: from Science Drivers to Reference Design and Anticipated Data Products
    (arXiv:0805.2366 [astro-ph])

    Parameters
    ---------------
    gamma: array (float)
      gamma parameter values
    mag: array (float)
      magnitude
    m5: array (float)
      fiveSigmaDepth values

    Returns
    -----------
    sigma_rand = (0.04-gamma)x-gammax**2
         with x = 10**(0.4*(mag-m5))

    """
    x = 10**(0.4*(mag-m5))
    return np.sqrt((0.04-gamma)*x+gamma*x**2)


def gamma(bands, exptime=30.):
    """
    Function to load and gamma vs m5 values
    end prepare for interpolation

    Parameters
    --------------
    bands: str
      bands of interest
    exptime: float, opt
      chosen exposure time

    Returns
    -----------
    dict of interpolators

    """
    # load the gamma file
    gamma = load('reference_files', 'gamma.hdf5')
    idx = np.abs(gamma['single_exptime']-30.) < 1.e-5
    selgamma = gamma[idx]

    # get interpolators for gamma and magflux
    gammadict = {}
    # magfluxdict = {}

    for b in bands:
        io = selgamma['band'] == b
        gammadict[b] = interp1d(
            selgamma[io]['mag'], selgamma[io]['gamma'], bounds_error=False, fill_value=0.)

    return gammadict


def load(theDir, fname):
    """
    Function to load LC data

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
    
    res = loopStack([searchname], objtype='astropyTable')
    
    return res
