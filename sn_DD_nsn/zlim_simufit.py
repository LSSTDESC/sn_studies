from sn_tools.sn_io import loopStack_params
from sn_tools.sn_utils import multiproc
import glob
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from scipy.interpolate import interp1d


def load(dbDir, dbName, snType):
    fullName = '{}/{}/*COSMOS*{}*.hdf5'.format(dbDir, dbName, snType)

    fis = glob.glob(fullName)

    print('hhh', len(fis))
    # tab = loopStack(fis, 'astropyTable')
    params = dict(zip(['objtype'], ['astropyTable']))

    tab = multiproc(fis, params, loopStack_params, 4)

    return tab


def zlim(tab):

    r = []
    for healpixID in np.unique(tab['healpixID']):
        idx = tab['healpixID'] == healpixID
        idx &= tab['fitstatus'] == 'fitok'
        sel = tab[idx]
        if len(sel) > 0:
            for season in np.unique(sel['season']):
                idxb = sel['season'] == season
                idxb &= np.sqrt(sel['Cov_colorcolor']) <= 0.04
                selb = sel[idxb]
                selb.sort(keys=['z'])
                if len(selb) >= 2:
                    norm = np.cumsum(selb['z'])[-1]
                    zlim = interp1d(
                        np.cumsum(selb['z'])/norm, selb['z'], bounds_error=False, fill_value=0.)
                    r.append((healpixID, season, zlim(0.95)))
                """
                fig, ax = plt.subplots()
                fig.suptitle(
                    'healpixID {} - season {}'.format(healpixID, season))
                n_bins = 20
                print(selb.columns)
                # plt.plot(sel['z'],np.sqrt(sel['Cov_colorcolor']),'ko')
                # plt.plot(sel['z'],np.cumsum(sel['z']),'ko')
                ax.plot(selb['z'], np.cumsum(selb['z'])/norm, 'ko')
                n, bins, patches = ax.hist(selb['z'], n_bins, density=True, histtype='step',
                                           cumulative=True, label='Empirical')
                print(n, bins)
                plt.show()
                """

    res = np.rec.fromrecords(r, names=['healpixID', 'season', 'zlim'])

    return res


def nsn(tab_all, zlims):

    r = []
    for healpixID in np.unique(tab_all['healpixID']):
        idx = tab_all['healpixID'] == healpixID
        idx &= tab_all['fitstatus'] == 'fitok'
        sel = tab_all[idx]
        if len(sel) > 0:
            for season in np.unique(sel['season']):
                idxb = sel['season'] == season
                idxb &= np.sqrt(sel['Cov_colorcolor']) <= 0.04
                selb = sel[idxb]
                io = zlims['healpixID'] == healpixID
                io &= zlims['season'] == season
                zlimit = zlims[io]
                if len(zlimit) > 0:
                    iko = selb['z'] <= zlimit['zlim'].item()
                    print(healpixID, season, len(selb), len(
                        selb[iko]), zlimit['zlim'].item())
                    r.append((healpixID, season, len(selb), len(
                        selb[iko]), zlimit['zlim'].item()))

    return np.rec.fromrecords(r, names=['healpixID', 'season', 'nsn_tot', 'nsn_zlim', 'zlim'])


parser = OptionParser(
    description='Estimate zlim from simulation+fit data')
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/users/gris/DD/Fit',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")


opts, args = parser.parse_args()

dbDir = opts.dbDir

dbName = opts.dbName

tab_faint = load(dbDir, dbName, 'faint')
tab_all = load(dbDir, dbName, 'all')

print(len(tab_faint), len(tab_all))
"""
fig, ax = plt.subplots()

ax.hist(tab['z'], histtype='step')

plt.show()
"""

res = zlim(tab_faint)

print(np.median(res['zlim']))
res_nsn = nsn(tab_all, res)
np.save('zlim_simufit.py', res)
np.save('nsn_simufit.py', res_nsn)
