from lsst.sims.speedObservatory import sky
from lsst.sims.speedObservatory import Telescope
import numpy as np
import csv
from lsst.sims.speedObservatory.utils import unix2mjd, mjd2djd
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import time
import os
import numpy.lib.recfunctions as rf


def nvisits_seasonlength(tab, nvisits=range(10, 300, 10)):
    """
    Function to estimate the max season length as a function of the number of visits

    Parameters
    ----------------
    tab: numpy array
      data to process
    nvisits: array, opt
      minimum number of visits per night (default: range(10,300,10))

    Returns
    ----------
    pandas df with two cols: nvisits, season_length
    """
    io = tab['season'] == 2
    sel = tab[io]
    r = []
    for nvisit in nvisits:
        idx = sel['nvisits'] >= nvisit
        idx &= sel['timeBegin']+sel['timeObs'] <= sel['nightEnd']
        selb = sel[idx]
        season_length = np.max(selb['night'])-np.min(selb['night'])
        r.append((nvisit, season_length))

    return pd.DataFrame(r, columns=['nvisits', 'season_length'])


def season(obs, season_gap=80., mjdCol='observationStartMJD'):
    """
    Function to estimate seasons

    Parameters
    --------------
    obs: numpy array
      array of observations
    season_gap: float, opt
       minimal gap required to define a season (default: 80 days)
    mjdCol: str, opt
      col name for MJD infos (default: observationStartMJD)

    Returns
    ----------
    original numpy array with season appended

    """

    # check wether season has already been estimated
    if 'season' in obs.dtype.names:
        return obs

    obs.sort(order=mjdCol)

    seasoncalc = np.ones(obs.size, dtype=int)
    if len(obs) > 1:
        diff = np.diff(obs[mjdCol])
        flag = np.where(diff > season_gap)[0]
        if len(flag) > 0:
            for i, indx in enumerate(flag):
                seasoncalc[indx+1:] = i+2

    obs = rf.append_fields(obs, 'season', seasoncalc)
    return obs


class Fields_up:
    """
    class to estimate the amount of time fields are observable per night
    with airmass <= airmass_max

    Parameters
    ---------------
    config_fields: csv file
      fields to consider (name, RA, Dec)
    airmass_max: float, opt
       max airmass for fields to be observable (default: 1.5)
    """

    def __init__(self, config_fields, airmass_max=1.5):

        self.tel = Telescope()
        self.fields = self.loadFields(config_fields)
        self.airmass_max = airmass_max
        print('fields', self.fields)

    def loadFields(self, config_file):
        """
        Method to load the fields

        Parameters
        ---------------
        config_file: csv file
          file with fields info 

        Returns
        -----------
        pandas df with cols name (of the field str), ra(rad), dec(rad)

        """
        tab = pd.read_csv(config_file, comment='#')
        radec = []
        for i, row in tab.iterrows():
            print(row['rah'], row['dech'])
            c = SkyCoord(row['rah'], row['dech'],
                         frame='icrs', unit=(u.hourangle, u.deg))
            radec.append((c.ra.radian, c.dec.radian))
        radec = np.array(radec)
        tab['ra'] = radec[:, 0]
        tab['dec'] = radec[:, 1]

        return tab

    def __call__(self, surveyStartTime, night_min, night_max):
        """
        Method where the processing is made

        Parameters
        ---------------
        surveyStartTime: int
           start time of the survey
        night_min: int
          min night number
        max_night: int
          max night number

        Returns
        -----------
        pandas df with infos

        """
        restot = pd.DataFrame()
        time_ref = time.time()
        for nightNum in range(night_min, night_max):
            res = self.processNight(surveyStartTime, nightNum)
            restot = pd.concat((restot, res))
        print('end of processing', time.time()-time_ref)
        return restot

    def processNight(self, surveyStartTime, nightNum):
        """
        Method to process a night

        Parameters
        ---------------
        surveyStartTime: int
           start time of the survey
        nightNum: int
          night number

        Returns
        -----------
        pandas df with infos

        """

        nightStart = sky.nightStart(surveyStartTime, nightNum)
        nightEnd = sky.nightEnd(surveyStartTime, nightNum)
        minRa = sky.raOfMeridian(nightStart)

        spacing = (nightEnd-nightStart)/100.

        ffields = self.fields.copy()
        res = pd.DataFrame()
        # for curTime in np.linspace(nightStart,nightEnd,10000):

        result_queue = multiprocessing.Queue()
        nproc = len(ffields)

        for j in range(nproc):
            fieldproc = ffields.iloc[j]
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.anaField_summary,
                                        args=(fieldproc, nightStart, nightEnd, nightNum, minRa, j, result_queue))
            p.start()

            resultdict = {}
        # get the results in a dict

        for i in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        res = pd.DataFrame()

        for key, vals in resultdict.items():
            res = pd.concat((res, vals))

        """
        res = ffields.groupby('name').apply(
            lambda x: self.anaField_summary(x, nightStart, nightEnd, nightNum, minRa)).reset_index()
        """
        return res

    def anaField_night(self, grp, nightStart, nightEnd, nightNum, minRa):
        """
        Method to estimate some infos per field and per nights

        Parameters
        ----------------
        grp: row array
          field to consider
        nightStart: int
           start night time
        nightEnd: int
           end night time
        nightNum: int
           night number
        minRa: float
          meridian Ra at the start of the night

        Returns
        -----------
        pandas df with the following cols:  mjd, ra, dec, alt, az, night, ha, curTime, airmass, time_meridian
        """

        r = []
        for curTime in np.arange(nightStart, nightEnd, 5.*60.):
            ra = grp['ra'].values.item()
            dec = grp['dec'].values.item()
            time_meridian = nightStart+(ra-minRa)*24.*3600./(2.*np.pi)
            alt, az = sky.radec2altaz(ra, dec, curTime)
            lst = sky.unix2lst(self.tel.longitude, curTime)
            ha = lst - ra
            airmass = 1./np.cos(np.pi/2-alt)
            if alt >= self.tel.minAlt and alt <= self.tel.maxAlt:
                if ha > np.pi:
                    ha = 2.*np.pi-ha
                r.append((unix2mjd(curTime), ra, dec, np.degrees(alt), np.degrees(
                    az), nightNum, ha, curTime, airmass, time_meridian))

        return pd.DataFrame(r, columns=[
            'mjd', 'ra', 'dec', 'alt', 'az', 'night', 'ha', 'curTime', 'airmass', 'time_meridian'])

    def anaField_summary(self, grp, nightStart, nightEnd, nightNum, minRa, j=0, output_q=None):
        """
        Method to estimate the amount of time of observation per night

        Parameters
        ----------------
        grp: row array
          field to consider
        nightStart: int
           start night time
        nightEnd: int
           end night time
        nightNum: int
           night number
        minRa: float
          meridian Ra at the start of the night

        Returns
        -----------
        pandas df with the following cols:  name,ra, dec, night,time_obs,nvisits,time_begin

        """
        ra = grp['ra']
        dec = grp['dec']
        name = grp['name']

        r = []
        for curTime in np.arange(nightStart, nightEnd, 5.*60.):
            time_meridian = nightStart+(ra-minRa)*24.*3600./(2.*np.pi)
            alt, az = sky.radec2altaz(ra, dec, curTime)
            lst = sky.unix2lst(self.tel.longitude, curTime)
            ha = lst - ra
            airmass = 1./np.cos(np.pi/2-alt)
            if alt >= self.tel.minAlt and alt <= self.tel.maxAlt and airmass <= self.airmass_max:
                if ha > np.pi:
                    ha = 2.*np.pi-ha
                r.append(curTime)

        time_obs = 0.0
        time_begin = 0.0
        nvisits = 0
        if len(r) > 0:
            time_obs = unix2mjd(np.max(r))-unix2mjd(np.min(r))
            nvisits = time_obs*24.*3600./30
            time_begin = unix2mjd(np.min(r))

        resdf = pd.DataFrame({'name': [name], 'ra': [ra], 'dec': [dec], 'night': [
                             nightNum], 'timeObs': [time_obs], 'nvisits': [nvisits],
            'timeBegin': [time_begin],
            'nightEnd': [unix2mjd(nightEnd)]})

        if output_q is not None:
            return output_q.put({j: resdf})
        else:
            return resdf


surveyStartTime = 1696118400  # october 1rst 2023
# surveyStartTime = 1664582461

night_min = 1
night_max = 365*10
airmass_max = 1.5

outName = 'fields_summary_{}_{}_{}.npy'.format(
    night_min, night_max, airmass_max)

if not os.path.isfile(outName):
    process = Fields_up('fields.csv', airmass_max=airmass_max)
    res = process(surveyStartTime, night_min, night_max)
    np.save(outName, res.to_records(index=False))

res = np.load(outName, allow_pickle=True)


# for name in np.unique(res['name']):

resdf = pd.DataFrame()
for name in np.unique(res['name']):
    idx = res['name'] == name
    idx &= res['nvisits'] > 0
    sel = season(np.copy(res[idx]),  mjdCol='night')
    rr = nvisits_seasonlength(sel)
    rr['name'] = name
    resdf = pd.concat((resdf, rr))

fig, ax = plt.subplots()
for name in np.unique(res['name']):
    idx = res['name'] == name
    sel = res[idx]
    ax.plot(sel['night'], sel['nvisits'], label=name)

ax.grid()
ax.legend()

fig, ax = plt.subplots()
rr = pd.DataFrame()
for name in np.unique(resdf['name']):
    idx = resdf['name'] == name
    sel = resdf[idx]
    ax.plot(sel['nvisits'], sel['season_length'], label=name)
    rr = pd.concat((rr, sel))

np.save('seasonlength_nvisits.npy', rr.to_records(index=False))


fontsize = 12
ax.set_xlabel('$N_{visits}$', fontsize=fontsize)
ax.set_ylabel('Max season length [days]', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

ax.grid()
ax.legend()


plt.show()
