from sn_tools.sn_io import loopStack_params
from sn_tools.sn_utils import multiproc
import glob
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from scipy.interpolate import interp1d

parser = OptionParser(
    description='Estimate zlim from simulation+fit data')
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/users/gris/DD/Fit/',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")


opts, args = parser.parse_args()

dbDir = opts.dbDir

dbName = opts.dbName

fullName = '{}/{}/*COSMOS*faint*.hdf5'.format(dbDir, dbName)

fis = glob.glob(fullName)

print('hhh', len(fis))
# tab = loopStack(fis, 'astropyTable')
params = dict(zip(['objtype'], ['astropyTable']))

tab = multiproc(fis, params, loopStack_params, 4)
print(len(tab))
"""
fig, ax = plt.subplots()

ax.hist(tab['z'], histtype='step')

plt.show()
"""
r = []
for healpixID in np.unique(tab['healpixID']):
    idx = tab['healpixID'] == healpixID
    idx &= tab['fitstatus'] == 'fitok'
    sel = tab[idx]
    if len(sel) > 0:
        for season in np.unique(sel['season']):
            # for season in [1]:
            idxb = sel['season'] == season
            idxb &= np.sqrt(sel['Cov_colorcolor']) <= 0.04
            selb = sel[idxb]
            selb.sort(keys=['z'])
            if len(selb) >= 2:
                norm = np.cumsum(selb['z'])[-1]
                zlim = interp1d(
                    np.cumsum(selb['z'])/norm, selb['z'], bounds_error=False, fill_value=0.)
                print('zlim', zlim(0.95))
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

print(np.median(res['zlim']))
