import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sn_tools.sn_obs import season as seasoncalc


def stat(grp):

    dictres = {}

    for ff in np.unique(grp['filter']):
        io = grp['filter'] == ff
        sel = grp[io]
        dictres['N_{}'.format(ff)] = [int(np.median(sel['numExposures']))]
        dictres['m5_{}'.format(ff)] = [np.median(sel['fiveSigmaDepth'])]

    dictres['nights'] = [len(np.unique(grp['night']))]

    grp = grp.sort_values(by=['observationStartMJD'])

    season_min = np.min(grp['observationStartMJD'])
    season_max = np.max(grp['observationStartMJD'])
    dictres['season_length'] = [season_max-season_min]
    nights = np.sort(grp['night'].unique())
    print(nights)
    cadence = 0.
    maxgap = 0.
    medgap = 0.
    if len(nights) >= 2:
        diff = np.asarray(nights[1:]-nights[:-1])
        cadence = np.median(diff).item()
        maxgap = np.max(diff).item()
        medgap = np.median(diff[diff > cadence]).item()

    dictres['cadence'] = [cadence]
    dictres['maxgap'] = [maxgap]
    dictres['medgap'] = [medgap]

    return pd.DataFrame(dictres)


def calc(grp):

    # coaddition per night
    gb = grp.groupby(['night', 'filter']).apply(
        lambda x: coadd(x)).reset_index()

    return gb


def coadd(grp):

    dictres = {}
    var_mean = ['pixRA', 'pixDec', 'observationStartMJD',
                'fieldRA', 'fieldDec', 'seeingFwhmEff', 'fiveSigmaDepth', 'season']
    var_sum = ['visitExposureTime', 'numExposures']

    for var in var_mean:
        dictres[var] = grp[var].mean()

    for var in var_sum:
        dictres[var] = grp[var].sum()

    # correct for 5-sigma depth
    dictres['fiveSigmaDepth'] += 1.25*np.log10(dictres['numExposures'])

    dictb = {}
    for key, vals in dictres.items():
        dictb[key] = []
        dictb[key].append(vals)

    return pd.DataFrame(dictb)


def load(dbDir, dbName, suffix, fieldName):

    fi = '{}/{}/{}_{}_{}.npy'.format(dbDir, dbName, dbName, suffix, fieldName)
    print(fi)
    res = np.load(fi, allow_pickle=True)
    return res


dbDir = '/media/philippe/LSSTStorage/ObsPixelized'
dbName = 'descddf_v1.4_10yrs'
fieldNames = ['COSMOS']
suffix = 'DD_nside_64_0.0_360.0_-1.0_-1.0'

for fieldName in fieldNames:
    tab = pd.DataFrame(load(dbDir, dbName, suffix, fieldName))
    print(tab.columns)
    npixels = len(np.unique(tab['healpixID']))
    print(npixels, len(tab))
    """
    plt.plot(tab['pixRA'], tab['pixDec'], 'ko')
    plt.show()
    """
    # df = tab.
    df = pd.DataFrame(np.copy(seasoncalc(tab.to_records(index=False))))
    dfcoadd = df.groupby(['healpixID']).apply(lambda x: calc(x)).reset_index()
    ii = dfcoadd['healpixID'] == 27237
    print(len(dfcoadd[ii]), dfcoadd[ii])
    finalres = dfcoadd.groupby(['healpixID', 'season']).apply(
        lambda x: stat(x)).reset_index()
    finalres = finalres.fillna(value=0)
    print(finalres)
    """
    for healpixID in np.unique(dfcoadd['healpixID']):
        ia = dfcoadd['healpixID'] == healpixID
        sel = dfcoadd[ia]
        print(healpixID, dfcoadd[[
              'night', 'filter', 'numExposures', 'visitExposureTime', 'fiveSigmaDepth']])
    """
    """
    for healpixID in np.unique(tab['healpixID']):
        ia = tab['healpixID'] == healpixID
        ib = df['healpixID'] == healpixID
        print(healpixID, len(tab[ia]), len(df[ib]))
        sel = df[ib]
        fig, ax = plt.subplots()
        for season in np.unique(sel['season']):
            ik = sel['season'] == season
            selb = sel[ik]
            ax.plot(selb['night'], selb['fiveSigmaDepth'], marker='o')
        plt.show()
    """
