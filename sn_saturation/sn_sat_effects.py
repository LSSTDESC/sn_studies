import h5py
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d

class LC:
    """
    class to load LC from h5py files

    """
    def __init__(self,dirFile, prodid):

        self.dirFile = dirFile
        self.prodid = prodid

        # load simulation parameters
        parName = 'Simu_{}.hdf5'.format(self.prodid)
        params = self.load_params('{}/{}'.format(self.dirFile, parName))
        # loop on this file using the simuPars list
        # select only LC with status=1
        ik = params['status'] == 1
        self.simuPars = params[ik]
        self.lcFile = '{}/LC_{}.hdf5'.format(self.dirFile, self.prodid)

    def load_params(self, paramFile):
        """
        Function to load simulation parameters

        Parameters
        ---------------
        paramFile: str
          name of the parameter file

        Returns
        -----------
        params: astropy table
        with simulation parameters

        """

        f = h5py.File(paramFile, 'r')
        print(f.keys(), len(f.keys()))
        params = Table()
        for i, key in enumerate(f.keys()):
            pars = Table.read(paramFile, path=key)
            params = vstack([params, pars])

        return params


    def getLC(self, index_hdf5):

         lc = Table.read(self.lcFile, path='lc_{}'.format(index_hdf5))

         return lc


class SaturationTime:
    """
    class to estimate saturation time vs z


    Parameters
    ---------------
    dirFile: str
      location directory wher LC are located
    x1: float
      SN x1
    color: float
      SN color
    nexp: int
      number of exposures
    exptime: float
      exposure time
   season: int
      season of observation
   psf_profile: str
      psf profile (default: single_gauss)

    """
    def __init__(self, dirFile, x1,color,nexp,exptime, season,psf_profile='single_gauss'):

        prodid = 'Saturation_{}_{}_{}_{}_{}_0'.format(nexp,exptime,x1,color,season)

        
        self.lc = LC(dirFile,prodid)

        pixel_npy = np.load('PSF_pixels_{}_summary.npy'.format(psf_profile))
        print(pixel_npy.dtype)
        print(pixel_npy['seeing'])
        print(pixel_npy['pixel_frac_med'])
         
        self.pixel_max = interp1d(pixel_npy['seeing'], pixel_npy['pixel_frac_med'], fill_value=0.0, bounds_error=False)

        seeings = dict(zip(['single_gauss', 'double_gauss'],
                                ['seeingFwhmEff', 'seeingFwhmGeom']))

        self.seeing_simu = seeings[psf_profile]
        print(self.lc.simuPars)


    def time_of_saturation(self, full_well):

        for par in self.lc.simuPars:
            lc = self.lc.getLC(par['index_hdf5'])
            print(lc.columns)
            lc.sort('time')
            lc['flux_e'] = lc['flux_e_sec']*lc['visitExposureTime']*self.pixel_max(lc[self.seeing_simu])
            print(lc[['flux_e','flux_e_sec','visitExposureTime','time']])
            
            
            break
        
        
