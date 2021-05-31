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

        outName = '{}/{}_{}_coadd.npy'.format(dirOut, dbName, fieldName)

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
                r_infos = [healpixID, season, z, T0, cadence_season]
                sel = self.cutoff(selobs, T0, z)
                r_infos += self.infos(sel, T0, z)

                r.append(r_infos)

        rr = None
        if len(r) > 0:
            rr = np.rec.fromrecords(r, names=[
                                    'healpixID', 'season', 'z', 'T0', 'cadence_season',
                                    'nepochs_bef', 'nepochs_aft',
                                    'cadence', 'cadence_std',
                                    'nphase_min', 'nphase_max', 'max_gap'])

        return rr

    def infos(self, obs, T0, z, phase_min=-10, phase_max=20):

        df = pd.DataFrame(np.copy(obs))
        dd = df.groupby(['night'])[self.mjdCol].median().reset_index()

        cadence = -1.0
        nphase_min = -1
        nphase_max = -1
        nepochs_bef = -1
        nepochs_aft = -1
        max_gap = -1.
        cadence_std = -1.
        if len(dd) >= 2:
            dd['phase'] = (T0-dd[self.mjdCol])/(1.+z)
            dd = dd.sort_values(by=[self.mjdCol])
            idx = dd[self.mjdCol]-T0 <= 0
            nepochs_bef = len(dd[idx])
            nepochs_aft = len(dd)-nepochs_bef
            idx = dd['phase'] <= phase_min
            nphase_min = len(dd[idx])
            idx = dd['phase'] >= phase_max
            nphase_max = len(dd[idx])
            diff_mjd = np.diff(dd[self.mjdCol])
            cadence = np.mean(diff_mjd)
            cadence_std = np.std(diff_mjd)
            max_gap = np.max(diff_mjd)

        r = [nepochs_bef, nepochs_aft, cadence, cadence_std,
             nphase_min, nphase_max, max_gap]
        return r

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


class Analysis:
    """
    """

    def __init__(self, data_in, dbName, fieldName,
                 nepochs_bef=4, nepochs_aft=10, nphase_min=1, nphase_max=1):

        self.nepochs_bef = nepochs_bef
        self.nepochs_aft = nepochs_aft
        self.nphase_min = nphase_min
        self.nphase_max = nphase_max

        # get the data
        self.fName = '{}_{}_slidingWindow.npy'.format(dbName, fieldName)
        if not os.path.isfile(self.fName):
            self.slidingWindow(data_in)

        data = np.load(self.fName, allow_pickle=True)

        res = pd.DataFrame(data).groupby(['healpixID', 'season']).apply(
            lambda x: self.effiObs(x)).reset_index()
        print(res['healpixID'].unique())

        for healpixID in res['healpixID'].unique():
            ida = res['healpixID'] == healpixID
            sel = res[ida]
            print(sel)
            fig, ax = plt.subplots()
            for season in sel['season'].unique():
                io = sel['season'] == season
                self.plot_indiv(ax, sel[io])

        plt.show()

    def slidingWindow(self, data):
        restot = None
        cad = ObsSlidingWindow()
        for healpixID in np.unique(data['healpixID']):
            idx = data['healpixID'] == healpixID
            rr = cad(data[idx], healpixID)
            if restot is None:
                restot = rr
            else:
                restot = np.concatenate((restot, rr))

        np.save(self.fName, restot)

    def effiObs(self, df):
        """
        Method to estimate observing efficiency vs z

        Returns
        -----------
        pandasdf with z, effi, var(effi) as columns
        """
        #df = pd.DataFrame(self.data.copy)

        # apply selection
        idx = df['nepochs_bef'] >= self.nepochs_bef
        idx &= df['nepochs_aft'] >= self.nepochs_aft
        idx &= df['nphase_min'] >= self.nphase_min
        idx &= df['nphase_max'] >= self.nphase_max

        df_sel = df[idx]

        # make groups (with z)
        group = df.groupby('z')
        group_sel = df_sel.groupby('z')

        # Take the ratio to get efficiencies
        rb = (group_sel.size()/group.size())
        #err = np.sqrt(rb*(1.-rb)/group.size())
        var = rb*(1.-rb)*group.size()

        rb = rb.array
        #err = err.array
        var = var.array

        rb[np.isnan(rb)] = 0.
        #err[np.isnan(err)] = 0.
        var[np.isnan(var)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             # 'effi_err': err,
                             'effi_var': var})

    def plot_indiv(self, ax, effis):
        """
        Method to plot efficiency vs redshift

        Parameters
        ---------------
        effis: pandas df
           data to plot
        """
        print('hello', effis)
        ax.errorbar(effis['z'], effis['effi'], yerr=np.sqrt(effis['effi_var']))


dirFiles = '../../ObsPixelized_128'
dbName = 'descddf_v1.5_10yrs'

fieldName = 'COSMOS'

data = Coadd(dirFiles, dbName, fieldName).data

data = season(data)
print(data.dtype)

restot = Analysis(data, dbName, fieldName).data

print(restot)

fig, ax = plt.subplots()

for season in np.unique(restot['season']):
    idx = restot['season'] == season
    ssel = restot[idx]
    print('season', season, np.unique(
        ssel['cadence_season']), np.median(ssel['cadence']))
    #ax.plot(ssel['z'], ssel['nepochs_bef'], 'ko')
    #ax.plot(ssel['z'], ssel['nepochs_aft'], 'ro')
    ax.plot(ssel['z'], ssel['max_gap'], 'ko')

    # break

plt.show()
