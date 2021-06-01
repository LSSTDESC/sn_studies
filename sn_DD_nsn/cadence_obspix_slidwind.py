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
                 phase_min=-10,phase_max=20,
                 blue_cutoff=380., red_cutoff=800., filterCol='filter', mjdCol='observationStartMJD'):

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
        do = ddf.groupby(['night'])[self.mjdCol].median().reset_index().to_records(index=False)

        cadence_season = np.mean(np.diff(do[self.mjdCol]))
        # loop on redshifts
        r = []
        zvals = list(np.arange(0., 1.10, 0.1))
        zvals[0] = 0.01

        self.infos_broad(selobs,season_min, season_max,zvals,healpixID,season)          
        
        for z in zvals:
            # for each redshift: get T0s

            T0_min = season_min-self.min_rf_qual*(1.+z)
            T0_max = season_max-self.max_rf_qual*(1.+z)
            
            sel = self.cutoff(selobs, T0s, z)
            print(sel)
            print(test)
            phases = (-do[self.mjdCol]+T0s[:, np.newaxis])/(1.+z)

            print(phases, phases.shape)
            print(np.min(phases, axis=1), np.max(phases, axis=1))
            flag = phases <= 0
            phases_neg = np.ma.array(phases, mask=~flag)
            phases_pos = np.ma.array(phases, mask=flag)
            print(np.min(phases, axis=1), np.max(phases, axis=1), phases_neg.count(axis=1),phases_pos.count(axis=1))
            print(test)

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

    def infos_broad(self, selobs,season_min, season_max, zvals,healpixID,season):

        # getting gen parameters
            
        T0s = np.arange(season_min, season_max, 3.)
        T0s = T0s.reshape((len(T0s),1))

        gen_par = np.rec.fromrecords(T0s,names=['T0'])
        gen_par = rf.append_fields(gen_par, 'min_rf_phase', [self.min_rf_phase]*len(T0s))
        gen_par = rf.append_fields(gen_par, 'max_rf_phase', [self.max_rf_phase]*len(T0s))
        #gen_par = np.tile(gen_par, (len(zvals),1))
        gen_par = np.repeat(gen_par,len(zvals))
        tt = np.tile(zvals, (len(T0s),1)).tolist()
        tt = list(itertools.chain(*tt))
        gen_par = rf.append_fields(gen_par, 'z', tt)

        # select T0s here depending on z
        idx = gen_par['T0']>= season_min-self.min_rf_qual*gen_par['z']
        idx &= gen_par['T0']<= season_max-self.max_rf_qual*gen_par['z']

        gen_par = np.copy(gen_par[idx])
        #print('hhh',gen_par)
    
        
        for b in 'i':
            night_neg, night_pos,phase_neg,phase_pos,z_vals,T0_vals = self.sel_band(b, selobs, gen_par)
            nepochs_bef = night_neg.count(axis=1)
            nepochs_aft = night_pos.count(axis=1)
            nphase_min = phase_neg.count(axis=1)
            nphase_max = phase_pos.count(axis=1)
            print(z_vals)
            print(T0_vals)
            print(nepochs_bef.shape,z_vals.shape,T0_vals.shape)
            res = np.rec.fromrecords([healpixID]*len(z_vals),names=['healpixID'])
            #print(res)
            res = rf.append_fields(res,'nepochs_bef',nepochs_bef)
            print(res)
                                 
        print(test)

    def sel_band(self, band,sel, gen_par):

        idx = sel['filter'] == band
        sel_obs = sel[idx]
        
        xi = (sel_obs[self.mjdCol]-gen_par['T0'][:, np.newaxis])
        p = xi/(1.+gen_par['z'][:,np.newaxis])
        
        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par['min_rf_phase'][:, np.newaxis]
        max_rf_phase = gen_par['max_rf_phase'][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)
        
        # remove LC points outside the (blue-red) range

        mean_restframe_wavelength = np.array(
            [self.telescope.mean_wavelength[band]]*len(sel_obs))
        mean_restframe_wavelength = np.tile(mean_restframe_wavelength, (len(gen_par), 1))/(1.+gen_par['z'][:, np.newaxis])
        #mean_restframe_wavelength = np.tile(mean_restframe_wavelength, (len(gen_par), 1))/(1.+z)
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff)

        flag_idx = np.argwhere(flag)
        phase_neg = p <= self.phase_min
        phase_pos = p >= self.phase_max
        night_neg = np.ma.array(np.tile(sel_obs['night'],(len(p),1)), mask=~(flag&phase_neg))
        print(night_neg.shape,p.shape,len(sel_obs),len(gen_par))
        night_pos = np.ma.array(np.tile(sel_obs['night'],(len(p),1)), mask=~(flag&phase_pos))
        phase_neg = np.ma.array(p, mask=~(flag&phase_neg))
        phase_pos= np.ma.array(p, mask=~(flag&phase_pos))
        z_vals = gen_par['z'][flag_idx[:, 0]]
        print('hhh',z_vals)
        
        T0_vals = gen_par['T0'][flag_idx[:, 0]]
        print(band, phase_neg.shape)
        #print(test)
        return night_neg, night_pos,phase_neg,phase_pos,z_vals,T0_vals
    
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
        obs_resh = obs.reshape(p.shape)
        
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

    def slidingWindow_multiproc(self, data, nproc=8):

        healpixIDs = np.unique(data['healpixID'])
        print('number of pixels', len(healpixIDs))
        params = {}
        params['data'] = data
        restot = multiproc(healpixIDs, params, self.slidingWindow, nproc=nproc)

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
            if restot is None:
                restot = rr
            else:
                #restot = np.concatenate((restot, rr))
                restot = np.hstack((restot, rr))
            print('Done with', healpixID, time.time()-time_refb)
        print('end of processing', j, time.time()-time_ref)
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

restot = Analysis(data, dbName, fieldName, nproc=1).data

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
