from sn_tools.sn_telescope import Telescope
import numpy as np
from scipy import interpolate


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

    m5_range = np.arange(20., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    f5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range))[:, 1]
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
    m5_range = np.arange(20., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    m5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range))[:, 1]
        m5_dict[band] = interpolate.interp1d(
            m5_range, flux_5, bounds_error=False, fill_value=0.0)

    return m5_dict
