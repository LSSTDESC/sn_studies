import numpy as np
import glob
import pandas as pd
import os
import numpy.lib.recfunctions as rf
import time
import itertools

from sn_tools.sn_obs import season
from sn_tools.sn_telescope import Telescope
import matplotlib.pyplot as plt
from sn_tools.sn_utils import multiproc
from sn_stackers.coadd_stacker import CoaddStacker


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

    def __init__(self, min_rf_phase=-20., max_rf_phase=60.,
                 min_rf_qual=-15., max_rf_qual=30.,
                 phase_min=-10, phase_max=20,
                 blue_cutoff=380., red_cutoff=800.,
                 filterCol='filter', mjdCol='observationStartMJD', nightCol='night',
                 T0step=3.):

        self.telescope = Telescope(airmass=1.2)
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase
        self.min_rf_qual = min_rf_qual
        self.max_rf_qual = max_rf_qual
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.filterCol = filterCol
        self.mjdCol = mjdCol
        self.nightCol = nightCol
        self.T0step = T0step

    def __call__(self, obs, healpixID):

        res = None
        seasons = np.unique(obs['season'])
        #seasons = [5]
        for season in seasons:

            pp = self.process(obs, season, healpixID)
            if pp is not None:
                if res is None:
                    res = pp
                else:
                    #print(res.dtype, pp.dtype)
                    res = np.concatenate((res, pp))
        return res

    def process(self, obs, season, healpixID, fast_processing=True):

        idx = obs['season'] == season

        selobs = obs[idx]

        # get min and max season
        season_min = np.min(selobs[self.mjdCol])
        season_max = np.max(selobs[self.mjdCol])

        selobs.sort(order=[self.mjdCol])
        ddf = pd.DataFrame(np.copy(selobs))
        do = ddf.groupby(['night'])[self.mjdCol].median(
        ).reset_index().to_records(index=False)

        cadence_season = np.mean(np.diff(do[self.mjdCol]))
        # loop on redshifts
        r = []
        zvals = list(np.arange(0., 1.10, 0.1))
        zvals[0] = 0.01

        # zvals = [0.01]
        time_ref = time.time()
        if fast_processing:
            res = self.infos_broad(selobs, season_min, season_max,
                                   zvals, healpixID, season, cadence_season)
            # print('end of processing ', season,
            #      healpixID, time.time()-time_ref)
        else:
            time_ref = time.time()
            resb = self.infos_loop(selobs, season_min, season_max,
                                   zvals, healpixID, season, cadence_season)
            # print('end of processing ', season,
            #      healpixID, time.time()-time_ref)
        #self.compare(res, resb)
        # print(test)

        return res

    def compare(self, resa, resb):

        print(resb)
        print('eee', resa)
        dfa = pd.DataFrame(resb)
        dfb = pd.DataFrame(resa)
        print('diff size', len(dfa)-len(dfb), len(dfa), len(dfb))
        # print(dfa.groupby(['z']).size(), dfb.groupby(['z']).size())

        dd = dfa.merge(dfb, left_on=['healpixID', 'season', 'z', 'T0'], right_on=[
            'healpixID', 'season', 'z', 'T0'])
        vvals = ['cadence_season', 'cadence', 'cadence_std',
                 'nepochs_bef', 'nepochs_aft', 'nphase_min', 'nphase_max', 'max_gap']

        for vv in vvals:
            vara = '{}_x'.format(vv)
            varb = vara.replace('_x', '_y')
            print(vv, np.mean(dd[vara]-dd[varb]))

        """
        wwhat = 'cadence'
        for i, val in dd.iterrows():
            print(i, val[['z', 'T0', '{}_x'.format(
                wwhat), '{}_y'.format(wwhat)]])
        """
        # print(test)

    def infos_loop(self, selobs, season_min, season_max,
                   zvals, healpixID, season, cadence_season):

        r = []
        for z in zvals:

            # for each redshift: get T0 s

            T0_min = season_min-self.min_rf_qual*(1.+z)
            T0_max = season_max-self.max_rf_qual*(1.+z)
            T0s = np.arange(T0_min, T0_max, self.T0step)

            for T0 in T0s:
                r_infos = [healpixID, season, z, T0, cadence_season]
                sel = self.cutoff(selobs, T0, z)
                # idx = sel[self.filterCol] == 'i'
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

    def infos_broad(self, selobs, season_min, season_max, zvals, healpixID, season, cadence_season):

        # getting gen parameters

        T0s = np.arange(season_min, season_max, self.T0step)
        gen_par = pd.DataFrame()
        for z in zvals:
            T0_min = season_min-self.min_rf_qual*(1.+z)
            T0_max = season_max-self.max_rf_qual*(1.+z)
            deltaT = T0_max-T0_min
            if deltaT >= self.T0step:
                T0vals = np.arange(T0_min, T0_max, self.T0step)
                # T0vals = [T0_min, T0_max]
                # T0vals = [60011.424383, 60014.424383]
                dd = pd.DataFrame(T0vals, columns=['T0'])
                dd['z'] = z
                gen_par = pd.concat((gen_par, dd))

            """
            print(type(T0vals))
            Tt = np.rec.fromrecords([z]*len(T0vals), names=['z'])
            print('oooo', Tt.shape, type(Tt))
            Tt = rf.append_fields(Tt, 'T0', T0vals)
            print(Tt)
            if gen_par is None:
                gen_par = Tt
            else:
                # T0s = np.concatenate((T0s, T0vals))
                gen_par = np.concatenate((gen_par, Tt))
            """
        if not gen_par.empty:
            gen_par = gen_par.to_records(index=False)
            gen_par = rf.append_fields(gen_par, 'min_rf_phase', [
                self.min_rf_phase]*len(gen_par))
            gen_par = rf.append_fields(gen_par, 'max_rf_phase', [
                self.max_rf_phase]*len(gen_par))
        else:
            return None
        """
        T0s = T0s.reshape((len(T0s), 1))

        gen_par = np.rec.fromrecords(T0s, names=['T0'])
        gen_par = rf.append_fields(gen_par, 'min_rf_phase', [
            self.min_rf_phase]*len(T0s))
        gen_par = rf.append_fields(gen_par, 'max_rf_phase', [
            self.max_rf_phase]*len(T0s))
        # gen_par = np.tile(gen_par, (len(zvals),1))
        gen_par = np.repeat(gen_par, len(zvals))
        tt = np.tile(zvals, (len(T0s), 1)).tolist()
        tt = list(itertools.chain(*tt))
        gen_par = rf.append_fields(gen_par, 'z', tt)

        print('before', gen_par['T0'])
        """
        """
        # select T0s here depending on z
        idx = gen_par['T0'] >= season_min-self.min_rf_qual*(1.+gen_par['z'])
        idx &= gen_par['T0'] <= season_max-self.max_rf_qual*(1.+gen_par['z'])

        print('hhhhh', season_min-self.min_rf_qual *
              (1.+gen_par['z']), season_max-self.max_rf_qual*(1.+gen_par['z']))
        gen_par = np.copy(gen_par[idx])
        print('hhh', gen_par)
        """
        night_neg, night_pos, phase_min, phase_max, z_vals, T0_vals, mjds = self.sel_band(
            selobs, gen_par)

        df = pd.DataFrame(mjds).stack().groupby(
            level=0).apply(lambda x: x.unique().tolist())

        cadmean = []
        cadstd = []
        maxgap = []
        """
        for vn in df:
            cm = 0.0
            cstd = 0.0
            mgp = 0.0
            if len(vn) >= 2:
                mydiff = np.diff(vn)
                if len(mydiff) >= 1:
                    cm = np.mean(mydiff)
                    cstd = np.std(mydiff)
                    mgp = np.max(mydiff)

            cadmean.append(cm)
            cadstd.append(cstd)
            maxgap.append(mgp)
        """
        #print(df, mjds)
        for bb in mjds:
            cm = 0.0
            cstd = 0.0
            mgp = 0.0
            unbb = np.unique(bb[~bb.mask])

            if len(unbb) >= 1:
                mydiff = np.diff(unbb)
                if len(mydiff) >= 1:
                    cm = np.mean(mydiff)
                    cstd = np.std(mydiff)
                    mgp = np.max(mydiff)

            cadmean.append(cm)
            cadstd.append(cstd)
            maxgap.append(np.float(mgp))

        #print('cadmean', cadmean)
        # nepochs_bef = night_neg.count(axis=1).tolist()
        # nepochs_aft = night_pos.count(axis=1).tolist()
        nepochs_bef = self.count_unique(night_neg)
        nepochs_aft = self.count_unique(night_pos)
        nphase_min = self.count_unique(phase_min)
        nphase_max = self.count_unique(phase_max)
        # nphase_min = phase_min.count(axis=1).tolist()
        # nphase_max = phase_max.count(axis=1).tolist()
        """
            cadence = np.diff(mjds, axis=1)
            cadmean = cadence.mean(axis=1, keepdims=True)
            cadstd = cadence.std(axis=1, keepdims=True)
            maxgap = cadence.max(axis=1, keepdims=True)
            """
        z = np.ma.median(z_vals, axis=1, keepdims=True).tolist()
        T0 = np.ma.median(T0_vals, axis=1, keepdims=True).tolist()
        # print(z)
        # print(T0)
        # print(z_vals.shape, z)
        # print('ooo', nepochs_bef.shape, z.shape, T0.shape)
        df = pd.DataFrame()
        df['z'] = list(itertools.chain(*z))
        df['T0'] = list(itertools.chain(*T0))
        df['nepochs_bef'] = nepochs_bef
        df['nepochs_aft'] = nepochs_aft
        df['nphase_min'] = nphase_min
        df['nphase_max'] = nphase_max
        df['healpixID'] = healpixID
        df['season'] = season
        df['cadence_season'] = cadence_season
        df['cadence'] = cadmean
        df['cadence_std'] = cadstd
        df['max_gap'] = maxgap
        """
            res = np.rec.fromrecords(
                list(itertools.chain(*z)), names=['z'])
            res = rf.append_fields(res, 'T0',  list(itertools.chain(*T0)))
            res = rf.append_fields(res, 'nepochs_bef', nepochs_bef)
            res = rf.append_fields(res, 'nepochs_aft', nepochs_aft)
            res = rf.append_fields(res, 'nphase_min', nphase_min)
            res = rf.append_fields(res, 'nphase_max', nphase_max)
            res = rf.append_fields(res, 'healpixID', [healpixID]*len(res))
            res = rf.append_fields(res, 'season', [season]*len(res))
            res = rf.append_fields(res, 'cadence_season', [
                cadence_season]*len(res))

            print(res)
            """
        return df.to_records(index=False)

    def count_unique(self, vv):

        df = pd.DataFrame(vv)

        return df.nunique(axis=1).to_list()

    def sel_band(self, sel, gen_par):
        """
        idx = sel['filter'] == band
        sel_obs = sel[idx]
        """
        sel_obs = sel
        # zs = np.tile(gen_par['z'], len(sel_obs))
        zs = np.repeat(gen_par['z'], len(sel_obs))
        zs = zs.reshape((len(gen_par['z']), len(sel_obs)))
        T0s = np.repeat(gen_par['T0'], len(sel_obs))
        T0s = T0s.reshape((len(gen_par['T0']), len(sel_obs)))

        xi = (sel_obs[self.mjdCol]-gen_par['T0'][:, np.newaxis])
        p = xi/(1.+gen_par['z'][:, np.newaxis])
        p = (sel_obs[self.mjdCol]-T0s)/(1.+zs)

        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
        max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)

        # remove LC points outside the (blue-red) range

        """
        mean_restframe_wavelength = np.array(
            [self.telescope.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+zs)
        """

        mean_restframe_wavelength = np.asarray(
            [self.telescope.mean_wavelength[obser[self.filterCol][-1]] for obser in sel_obs])
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1))/(1.+zs)
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        flag_idx = np.argwhere(flag)
        phase_neg = p <= 0
        phase_pos = p > 0
        pha_min = p <= self.phase_min
        pha_max = p >= self.phase_max
        night_neg = np.ma.array(
            np.tile(sel_obs[self.nightCol], (len(p), 1)), mask=~(flag & phase_neg))
        night_pos = np.ma.array(
            np.tile(sel_obs[self.nightCol], (len(p), 1)), mask=~(flag & phase_pos))
        mjds = np.ma.array(
            np.tile(sel_obs[self.nightCol], (len(p), 1)), mask=~flag)
        phase_min = np.ma.array(p, mask=~(flag & pha_min))
        phase_max = np.ma.array(p, mask=~(flag & pha_max))
        # z_vals = gen_par['z'][flag_idx[:, 0]]
        z_vals = np.ma.array(zs)
        T0_vals = np.ma.array(T0s)

        # print(p, mean_restframe_wavelength)
        return night_neg, night_pos, phase_min, phase_max, z_vals, T0_vals, mjds

    def infos(self, obs, T0, z):

        df = pd.DataFrame(np.copy(obs))
        df['phase'] = (df[self.mjdCol]-T0)/(1.+z)
        dd = df.groupby(['night'])[self.mjdCol].median().reset_index()

        cadence = 0.0
        nphase_min = 0
        nphase_max = 0
        nepochs_bef = 0
        nepochs_aft = 0
        max_gap = 0.0
        cadence_std = 0.0
        if len(dd) >= 1:
            dd['phase'] = (dd[self.mjdCol]-T0)/(1.+z)
            dd = dd.sort_values(by=[self.mjdCol])
            idx = dd['phase'] <= 0
            nepochs_bef = len(dd[idx])
            nepochs_aft = len(dd)-nepochs_bef
            idx = df['phase'] <= self.phase_min
            nphase_min = len(df[idx])
            idx = df['phase'] >= self.phase_max
            nphase_max = len(df[idx])
        if len(dd) >= 2:
            diff_mjd = np.diff(dd[self.nightCol])
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
                 nepochs_bef=4, nepochs_aft=10,
                 nphase_min=1, nphase_max=1,
                 nproc=8):

        self.nepochs_bef = nepochs_bef
        self.nepochs_aft = nepochs_aft
        self.nphase_min = nphase_min
        self.nphase_max = nphase_max

        # get the data
        self.fName = '{}_{}_slidingWindow.npy'.format(dbName, fieldName)
        if not os.path.isfile(self.fName):
            self.slidingWindow_multiproc(data_in, nproc=nproc)

        data = np.load(self.fName, allow_pickle=True)

        print('boo', type(data))
        dd = pd.DataFrame(np.copy(data))
        res = dd.groupby(['healpixID', 'season']).apply(
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

    def slidingWindow_multiproc(self, data, nproc=8):

        healpixIDs = np.unique(data['healpixID'])
        # print(healpixIDs)
        #healpixIDs = [108951, 108953]
        print('number of pixels', len(healpixIDs))
        params = {}
        params['data'] = data
        nproc = np.min([len(healpixIDs), nproc])
        restot = multiproc(healpixIDs, params, self.slidingWindow, nproc=nproc)
        print('hello', type(restot), restot.dtype)
        np.save(self.fName, restot)

    def slidingWindow(self, healpixIDs, params={}, j=0, output_q=None):
        print('processing', j, len(healpixIDs))
        time_ref = time.time()

        restot = None
        cad = ObsSlidingWindow()
        data = params['data']
        for healpixID in healpixIDs:
            time_refb = time.time()
            idx = data['healpixID'] == healpixID
            rr = cad(data[idx], healpixID)
            if rr is not None:
                if restot is None:
                    restot = rr
                else:
                    #print('hello', i, healpixID, len(restot), len(rr))
                    restot = np.concatenate((restot, rr))
                #restot = np.hstack((restot, rr))
            #print('Done with', healpixID, time.time()-time_refb)
        print('end of processing', j, time.time()-time_ref, len(restot))
        if output_q is not None:
            return output_q.put({j: restot})
        else:
            return restot

    def effiObs(self, df):
        """
        Method to estimate observing efficiency vs z

        Returns
        -----------
        pandasdf with z, effi, var(effi) as columns
        """
        # df = pd.DataFrame(self.data.copy)

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
        # err = np.sqrt(rb*(1.-rb)/group.size())
        var = rb*(1.-rb)*group.size()

        rb = rb.array
        # err = err.array
        var = var.array

        rb[np.isnan(rb)] = 0.
        # err[np.isnan(err)] = 0.
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

restot = Analysis(data, dbName, fieldName, nproc=4).data

print(restot)

fig, ax = plt.subplots()

for season in np.unique(restot['season']):
    idx = restot['season'] == season
    ssel = restot[idx]
    print('season', season, np.unique(
        ssel['cadence_season']), np.median(ssel['cadence']))
    # ax.plot(ssel['z'], ssel['nepochs_bef'], 'ko')
    # ax.plot(ssel['z'], ssel['nepochs_aft'], 'ro')
    ax.plot(ssel['z'], ssel['max_gap'], 'ko')

    # break

plt.show()
