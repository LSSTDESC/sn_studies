from sn_tools.sn_io import loopStack
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

fullName = '{}/{}/*COSMOS*faint*.hdf5'.format(dbDir,dbName)

fis = glob.glob(fullName)

print('hhh',len(fis))
tab = loopStack(fis,'astropyTable')

print(len(tab))

for healpixID in np.unique(tab['healpixID']):
    idx = tab['healpixID']==healpixID
    idx &= tab['fitstatus']=='fitok'
    sel = tab[idx]
    if len(sel)>0:
        for season in np.unique(sel['season']):
            idxb = sel['season']==season
            idxb &= np.sqrt(sel['Cov_colorcolor'])<=0.04
            selb = sel[idxb]
            selb.sort(keys=['z'])
            if len(selb)>0:
                zlim = interp1d(np.cumsum(sel['z']), sel['z'], bounds_error=False, fill_value=0.)
                print('zlim',zlim(0.95))
                fig, ax = plt.subplots()
                n_bins=20
                print(selb.columns)
                #plt.plot(sel['z'],np.sqrt(sel['Cov_colorcolor']),'ko')
                #plt.plot(sel['z'],np.cumsum(sel['z']),'ko')
                n, bins, patches = ax.hist(selb['z'], n_bins, density=True, histtype='step',
                                           cumulative=True, label='Empirical')
                print(n,bins)
                plt.show()
