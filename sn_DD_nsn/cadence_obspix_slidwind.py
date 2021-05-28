import numpy as np
import glob
from sn_stackers.coadd_stacker import CoaddStacker
import pandas as pd
import os
from sn_tools.sn_obs import season
from sn_tools.sn_telescope import Telescope
import matplotlib.pyplot as plt


class Coadd:
    """
    class to perform a coadd od obs per healpixID, night, filter

    Parameters
    ---------------
    dirFiles: str
      location dir of the files to process
    dbName: str
       name of the db (OS) to process
    fieldName: str
       DD field to process
    dirOut: str, opt
       outputDirectory for the results
    """

    def __init__(self, dirFiles, dbName, fieldName, dirOut='.'):

        outName = '{}/{}_coadd.npy'.format(dirOut, dbName)

        if not os.path.isfile(outName):
            # load data
            res = self.load(dirFiles, dbName, fieldName)
            df = pd.DataFrame(res)

            obs = df.groupby(['healpixID', 'night', 'filter']).apply(
                lambda x: agg(x)).reset_index()
            for vv in ['healpixID', 'night', 'numExposures']:
                obs[vv] = obs[vv].astype(int)
            obs = obs.to_records(index=False)
            obs = season(obs)
            np.save(outName, obs)

        self.data = np.load(outName, allow_pickle=True)

    def agg(self, grp):
        """
        Method to perfom some ops on a grp df

        Parameters
        ---------------
        grp: pandas df grp
          group to process

        Returns
        -----------
        pandas df with median and sum values of the initial group

        """
        meds = ['observationId', 'pixRA', 'pixDec', 'observationStartMJD',
                'fieldRA', 'fieldDec', 'seeingFwhmEff', 'fiveSigmaDepth']

        sums = ['visitExposureTime', 'numExposures']

        dictres = {}

        for med in meds:
            dictres[med] = [grp[med].median()]

        for su in sums:
            dictres[su] = [grp[su].sum()]

        dictres['fiveSigmaDepth'] += 1.25*np.log10(dictres['numExposures'])

        return pd.DataFrame(dictres)

    def load(self, dirFile, dbName, fieldName):

        fis = glob.glob('{}/{}/*{}*'.format(dirFiles, dbName, fieldName))

        res = None
        for fi in fis:
            tt = np.load(fi, allow_pickle=True)
        if res is None:
            res = tt
        else:
            res = np.concatenate((res, tt))


class ObsSlidingWindow:
    """

    """

    def __init__(self, min_rf_phase=-20., max_rf_phase=60., min_rf_qual=-15., max_rf_qual=30.,
                 blue_cutoff=380., red_cutoff=800., filterCol='filter', mjdCol='observationStartMJD'):

        self.telescope = Telescope(airmass=1.2)
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase
        self.min_rf_qual = min_rf_qual
        self.max_rf_qual = max_rf_qual
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.filterCol = filterCol
        self.mjdCol = mjdCol

    def __call__(self, obs, healpixID):

        res = None
        for season in np.unique(obs['season']):
            pp = self.process(obs, season, healpixID)
            if pp is not None:
                if res is None:
                    res = pp
                else:
                    res = np.concatenate((res, pp))
        return res

    def process(self, obs, season, healpixID):

        idx = obs['season'] == season
        selobs = obs[idx]

        # get min and max season
        season_min = np.min(selobs[self.mjdCol])
        season_max = np.max(selobs[self.mjdCol])

        selobs.sort(order=[self.mjdCol])
        ddf = pd.DataFrame(np.copy(selobs))
        do = ddf.groupby(['night'])[self.mjdCol].median().reset_index()
        print(do.columns)
        cadence_season = np.mean(np.diff(do[self.mjdCol]))
        # loop on redshifts
        r = []
        zvals = list(np.arange(0., 1.10, 0.1))
        zvals[0] = 0.01
        for z in zvals:
            # for each redshift: get T0s
            T0_min = season_min-self.min_rf_qual*(1.+z)
            T0_max = season_max-self.max_rf_qual*(1.+z)
            for T0 in np.arange(T0_min, T0_max, 3.):
                sel = self.cutoff(selobs, T0, z)
                nepochs_bef, nepochs_aft, cadence = self.infos(sel, T0)
                print(z, T0, len(sel), nepochs_bef, nepochs_aft, cadence)
                r.append((healpixID, season, z, T0,
                          nepochs_bef, nepochs_aft, cadence, cadence_season))

        rr = None
        if len(r) > 0:
            rr = np.rec.fromrecords(r, names=[
                                    'healpixID', 'season', 'z', 'T0', 'nepochs_bef', 'nepochs_aft', 'cadence', 'cadence_season'])

        return rr

    def infos(self, obs, T0):

        df = pd.DataFrame(np.copy(obs))
        dd = df.groupby(['night'])[self.mjdCol].median().reset_index()

        dd = dd.sort_values(by=[self.mjdCol])
        idx = dd[self.mjdCol]-T0 <= 0
        nepochs_bef = len(dd[idx])
        nepochs_aft = len(dd)-nepochs_bef
        cadence = -1.0
        if len(dd) >= 2:
            cadence = np.mean(np.diff(dd[self.mjdCol]))

        return nepochs_bef, nepochs_aft, cadence

    def cutoff(self, obs, T0, z):
        """ select observations depending on phases

        Parameters
        -------------
        obs: array
          array of observations
        T0: float
          daymax of the supernova
        z: float
          redshift

        Returns
        ----------
        array of obs passing the selection
        """

        mean_restframe_wavelength = np.asarray(
            [self.telescope.mean_wavelength[obser[self.filterCol][-1]] /
             (1. + z) for obser in obs])

        p = (obs[self.mjdCol]-T0)/(1.+z)

        idx = (p >= 1.000000001*self.min_rf_phase) & (p <=
                                                      1.00001*self.max_rf_phase)
        idx &= (mean_restframe_wavelength > self.blue_cutoff)
        idx &= (mean_restframe_wavelength < self.red_cutoff)
        return obs[idx]


dirFiles = '../../ObsPixelized_128'
dbName = 'descddf_v1.5_10yrs'

fieldName = 'COSMOS'

data = Coadd(dirFiles, dbName, fieldName).data

data = season(data)
print(data.dtype)

restot = None
cad = ObsSlidingWindow()
for healpixID in np.unique(data['healpixID']):
    idx = data['healpixID'] == healpixID
    rr = cad(data[idx], healpixID)
    if restot is None:
        restot = rr
    else:
        restot = np.concatenate((restot, rr))
    break

print(restot)

fig, ax = plt.subplots()

for season in np.unique(restot['season']):
    idx = restot['season'] == season
    ssel = restot[idx]
    print('season', season, np.unique(
        ssel['cadence_season']), np.median(ssel['cadence']))
    ax.plot(ssel['z'], ssel['nepochs_bef'], 'ko')
    ax.plot(ssel['z'], ssel['nepochs_aft'], 'ro')

    # break

plt.show()
