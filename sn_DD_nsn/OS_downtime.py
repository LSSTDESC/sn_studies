import numpy as np
import pandas as pd
from optparse import OptionParser
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import ephem
import os

def unix2mjd(timestamp):
    """Convert from unix timestamp to MJD
    Parameters
    ----------
    timestamp : float
        The unix timestamp to be converted.
    Returns
    -------
    The modified Julian date corresponding to `timestamp`
    Notes
    -----
    This does the crudest possible conversion, and does not take into
    account leap seconds (or potentially other discrepancies I don't know about)
    """
    return timestamp / 86400 + 40587


def mjd2unix(mjd):
    """Convert from MJD to unix timestamp
    Parameters
    ----------
    mjd : float
        The modified Julian date to be converted.
    Returns
    -------
    The unix timestamp corresponding to `mjd`.
    Notes
    -----
    This does the crudest possible conversion, and does not take into
    account leap seconds (or potentially other discrepancies I don't know about)
    """
    return (mjd - 40587) * 86400


def mjd2djd(mjd):
    """Convert from modified Julian date to Dublin Julian date
    pyephem uses Dublin Julian dates
    Parameters
    ----------
    mjd : float
        The modified Julian date to be converted.
    Returns
    -------
    The Dublin Julian date corresponding to `mjd`.
    """
    # (this function adapted from Peter Yoachim's code)
    doff = 15019.5  # this equals ephem.Date(0)-ephem.Date('1858/11/17')
    return mjd - doff


def djd2mjd(djd):
    """Convert from Dublin Julian date to Modified Julian date
    pyephem uses Dublin Julian dates
    Parameters
    ----------
    djd : float
        The Dublin Julian date to be converted.
    Returns
    -------
    The modified Julian date corresponding to `djd`.
    """
    doff = 15019.5  # this equals ephem.Date(0)-ephem.Date('1858/11/17')
    return djd + doff


def all_nights(dbDir, dbName, dbExtens):
    """
    Function to get obs for all the nights (including nights with no obs!)

    Parameters
    --------------
    dbDir: str
      location dir of the file
    dbName: str
      name of the OS to consider
    dbExtens: str
       extension of the file to process

    Returns
    ----------
    observations for all the nights

    """
    fullName = '{}/{}.{}'.format(dbDir, dbName, dbExtens)

    df = pd.DataFrame(np.load(fullName, allow_pickle=True))

    # get moon* columns
    r = []
    for bb in df.columns:
        if 'moon' in bb:
            r.append(bb)

    rb = r+['mjd']

    # get medians here

    meds = df.groupby(['night'])[rb].median().reset_index()

    print(meds)

    # now estimate all the nights
    night_min = meds['night'].min()
    night_max = meds['night'].max()

    meds = meds.sort_values(by=['night'])
    mjd_interp = interp1d(meds['night'], meds['mjd'],
                          bounds_error=False, fill_value=0.)

    nights_all = range(night_min, night_max+1)
    mjds_all = mjd_interp(nights_all)
    nights = pd.DataFrame(nights_all, columns=['night'])
    nights['mjd'] = mjds_all
    nights['downtime'] = 0
    idx = nights['night'].isin(meds['night'])
    nights.loc[~idx, 'downtime'] = 1

    interp = interpol(r, meds)

    # now add moon values to night_df(from interp)
    for vv in r:
        nights[vv] = interp[vv](nights['night'])

    nights['moonPhase'] = getMoon(nights['night'], mjd_interp)
    return nights, interp, mjd_interp


def getMoon(nights, mjd_interp):
    """
    Function to estimate the moonPhase

    Parameters
    --------------
    nights: list(int)
      list of nights to process
    mjd_interp: interpolator
      to convert a night to an mjd

    Returns
    ----------
    list of moon phases

    """
    """
    gatech = ephem.Observer()
    Rubin_lat = np.rad2deg(-0.527868529)
    Rubin_lon = np.rad2deg(-1.2348102646986)
    gatech.lon, gatech.lat = str(Rubin_lon),str(Rubin_lat)
    moon = ephem.Moon()
    """
    #d = ephem.Date(nights['mjd'])
    """
    print(nights.dtypes)
    nights['mjd'] = nights['mjd'].apply(str)
    print(nights['mjd'])
    print(nights.dtypes)
    """

    r = []
    for night in nights:
        mjd = mjd_interp(night)
        moon = ephem.Moon(mjd2djd(mjd))
        r.append(moon.phase)

    return r


def interpol(r, meds):
    """
    Function to make interpolations on a set of observations

    Parameters
    --------------
    r: list(str)
      list of vars to interpolate
    meds: pandas df
       median values per night

    Returns
    ---------
    interpolator dict

    """
    interp = {}
    for vv in r:
        interp[vv] = interp1d(meds['night'], meds[vv],
                              bounds_error=False, fill_value=0.)

    return interp


def load_DD(fieldDir, nside, dbName, fieldName):
    """
    Function to load pixels corresponding to DDFs

    Parameters
    --------------
    fieldDir: str
       location directory of the files
    nside: int
      nside parameter for pixellisation of the sky
    dbName: str
      name of the OS to process
    fieldName: str
      name of the DD field to consider

    Returns
    ----------
    numpy array with pixel information

    """
    fullName = '{}/ObsPixelized_{}_{}_{}_night.npy'.format(
        fieldDir, nside, dbName, fieldName)
    tab = np.load(fullName, allow_pickle=True)

    return np.copy(tab)


def filterseq(grp):

    seq = ''.join(sorted(grp['filter'].tolist()))

    return pd.DataFrame({'filterseq': [seq], 'mjd': [np.round(grp['night'].median(), 1)]})


def selectSeason(DD_field, nights, season=1):
    """
    Function to select obs corresponding to a season

    Parameters
    --------------
    DD_fields: array
      data to consider
    nights: array
      list of nights
    season: int, opt
      season number (default: 1)

    Returns
    ----------
    night_sel: array of all the nights corresponding to season
    sel_DD: DD observations corresponding to season

    """
    idx = DD_field['season'] == season
    sel_DD = DD_field[idx]
    night_min = sel_DD['night'].min()
    night_max = sel_DD['night'].max()

    ido = nights['night'] >= night_min
    ido &= nights['night'] <= night_max
    nights_sel = nights[ido]

    return nights_sel, sel_DD


def plot(DD_field, nights, season, whatx='night', whaty='moonPhase'):
    """
    Function to plot whaty vs whatx

    Parameters
    ---------------
    DD_field: pandas df
       df with DD infos (night of observations, ...)
    nights: pandas df
      df with nights infos (on/off, ...)
    season: int
      season number to process
    whatx: str, opt
      x var to plot (default: night)
    whaty: str, opt
      y var to plot (default: moonPhase)

    """
    # now make some plots
    fig, ax = plt.subplots()

    # select data for season=season
    nights_sel, DD_sel = selectSeason(DD_field, nights, season)

    idx = nights_sel['downtime'] == 0
    nights_on = nights_sel[idx]
    nights_off = nights_sel[~idx]

    ax.plot(nights_on[whatx], nights_on[whaty], 'ko', mfc='None')
    ax.plot(nights_off[whatx], nights_off[whaty], 'ro', mfc='None')

    DD_seq = DD_sel.groupby(['healpixID', 'night']).apply(
        lambda x: filterseq(x)).reset_index()

    rrec = DD_seq.to_records(index=False)
    rrec = pd.DataFrame(np.unique(rrec[['night', 'filterseq']]))
    rrec['moonPhase'] = getMoon(rrec['night'], mjd_interp)
    ax.plot(rrec[whatx], rrec[whaty], 'ko', mfc='k')
    # plot with sequences of filter
    sequences = ['gir', 'yz', 'giryz', 'g', 'giruy', 'y']
    mc = ['b*', 'm*', 'm*', 'g*', 'g*', 'y*']
    hh = dict(zip(sequences, mc))
    for key, vals in hh.items():
        idx = rrec['filterseq'] == key
        ax.plot(rrec[idx][whatx], rrec[idx][whaty], vals)


def statSeason(dbName, DD_field, nights, season):
    """
    Function to estimate stat

    Parameters
    ---------------
    dbName: str
      OS name of consideration
    DD_field: pandas df
       df with DD infos (night of observations, ...)
    nights: pandas df
      df with nights infos (on/off, ...)
    season: int
      season number to process

    Returns
    -----------
    pandas df with stat

    """
    # put the results in a dict
    stat = {}

    # select data for season=season
    nights_sel, DD_sel = selectSeason(DD_field, nights, season)

    idx = nights_sel['downtime'] == 0
    nights_on = nights_sel[idx]
    nights_off = nights_sel[~idx]
    stat['nights_on'] = [len(nights_on)]
    stat['nights_off'] = [len(nights_off)]
    stat['nights_all'] = [len(nights_sel)]

    DD_seq = DD_sel.groupby(['healpixID', 'night']).apply(
        lambda x: filterseq(x)).reset_index()
    rrec = DD_seq.to_records(index=False)
    rrec = pd.DataFrame(np.unique(rrec[['night', 'filterseq']]))
    sequences = ['gir', 'yz', 'giryz', 'g', 'giruy', 'g', 'r', 'i', 'z', 'y']
    nights_ddf = []
    for key in sequences:
        idx = rrec['filterseq'] == key
        nights_ddf += rrec[idx]['night'].to_list()
        stat['nights_{}'.format(key)] = [len(rrec[idx])]
    nights_ddf.sort()
    iddx = nights_on['night'].isin(nights_ddf)
    stat['nights_on_noddf'] = [len(nights_on[~iddx])]
    rrec = rrec.sort_values(by=['night'])
    if 'descddf' in dbName:
        io = rrec['filterseq'] == 'yz'
        rrec = rrec[io]

    cadence = np.median(
        rrec['night'][1:].values-rrec['night'][:-1].values)

    stat['cadence'] = [cadence]

    return pd.DataFrame.from_dict(stat)


def analysis(nights, fieldDir, nside, dbName, fieldName):
    """
    Function to analyse a field

    Parameters
    ---------------
    nights: pandas df
      df with night infos
    fieldDir: str
      location dir of the file to process
    nside: int
      healpix param nside
    dbName: str
      OS name
    fieldName: str
      name of the field to process

    Returns
    -----------
    pandas df with stat infos

    """
    # get DD fields
    DD_field = pd.DataFrame(
        load_DD(fieldDir, nside, dbName, fieldName))
    # add vars here

    for key, vals in interp.items():
        DD_field[key] = vals(DD_field['night'])

        #season = 1
        # for season in range(1,10):
        #plot(DD_field, nights, season)
    dftot = pd.DataFrame()
    for season in range(1, 11):
        print('season', season)
        df = statSeason(dbName, DD_field, nights, season)
        df['fieldName'] = fieldName
        df['dbName'] = dbName
        df['season'] = season
        dftot = pd.concat((dftot, df))

    return dftot


parser = OptionParser()

parser.add_option("--dbDir", type=str, default='../DB_Files',
                  help="OS dir location[%default]")
parser.add_option("--dbName", type=str, default='descddf_v1.5_10yrs',
                  help="OS name[%default]")
parser.add_option("--dbExtens", type=str, default='npy',
                  help="OS extens (db or npy) [%default]")
parser.add_option("--fieldNames", type=str, default='COSMOS',
                  help="field to consider for this study  [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="healpix nside [%default]")
parser.add_option("--fieldDir", type=str, default='.',
                  help="dir where the field file is  [%default]")
parser.add_option('--outputDir', type=str, default='/sps/lsst/users/gris/OS_downtime',
                  help='output directory [%default]')

opts, args = parser.parse_args()

# get infos for all the nights

dbDir = opts.dbDir
dbName = opts.dbName
dbExtens = opts.dbExtens
nside = opts.nside
fieldDir = opts.fieldDir
fieldNames = opts.fieldNames
outputDir = opts.outputDir

nights, interp, mjd_interp = all_nights(dbDir, dbName, dbExtens)

fieldNames = fieldNames.split(',')

df = pd.DataFrame()

for fieldName in fieldNames:
    print('processing', fieldName)
    res = analysis(nights, fieldDir, nside, dbName, fieldName)
    df = pd.concat((df, res))

print(df)

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

outName = '{}/{}.csv'.format(outputDir,dbName)
df.to_csv(outName, index=False)
