import numpy as np
from optparse import OptionParser
import glob
from sn_tools.sn_io import loopStack
import pandas as pd
import matplotlib.pyplot as plt


def loadMetricResults(dbDir, dbName, fieldName):

    dirName = '{}/{}/NSN_{}'.format(dbDir, dbName, fieldName)

    fis = glob.glob('{}/*.hdf5'.format(dirName))

    print(fis)

    zlim_metric = loopStack(fis, 'astropyTable').to_pandas()

    for vv in ['healpixID', 'season']:
        zlim_metric[vv] = zlim_metric[vv].astype(int)

    print(zlim_metric.columns)
    zlim_metric = zlim_metric[['healpixID',
                               'season', 'zlim_faint', 'nsn_med_faint']]

    return zlim_metric


parser = OptionParser(
    description='Compare zlim from simulation+fit data and metric')
parser.add_option("--dbDir", type="str",
                  default='/sps/lsst/users/gris/MetricOutput_DD_new_128_epochs_4_10/',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")
parser.add_option("--fieldName", type="str",
                  default='COSMOS',
                  help="DD field to consider [%default]")
parser.add_option("--simufitName", type="str",
                  default='zlim_simufit.py.npy',
                  help="zlim result from simulation+fit [%default]")
parser.add_option("--nsn_simufitName", type="str",
                  default='nsn_simufit.py.npy',
                  help="nsn result from simulation+fit [%default]")


opts, args = parser.parse_args()

zlim_simufit = pd.DataFrame(np.load(opts.simufitName, allow_pickle=True))
nsn_simufit = pd.DataFrame(np.load(opts.nsn_simufitName, allow_pickle=True))

zlim_metric = loadMetricResults(opts.dbDir, opts.dbName, opts.fieldName)
print(zlim_metric)
print(zlim_simufit)

zlim_merge = zlim_metric.merge(zlim_simufit, left_on=[
                               'healpixID', 'season'],  right_on=['healpixID', 'season'])

nsn_merge = zlim_metric.merge(nsn_simufit, left_on=[
    'healpixID', 'season'],  right_on=['healpixID', 'season'])


print(zlim_merge)

fig, ax = plt.subplots()
diff = zlim_merge['zlim_faint']-zlim_merge['zlim']
ax.hist(diff, histtype='step')

print(np.mean(diff), np.std(diff))
print(nsn_merge.columns)
figb, axb = plt.subplots()
norm = 105.*0.96
diff = (nsn_merge['nsn_med_faint']-nsn_merge['nsn_zlim']/norm) / \
    nsn_merge['nsn_med_faint']
axb.hist(diff, histtype='step', bins=30)
#axb.plot(nsn_merge['healpixID'], diff, 'ko')
print(np.mean(diff), np.std(diff), np.sum(
    nsn_merge['nsn_med_faint']), np.sum(nsn_merge['nsn_zlim']/norm))

plt.show()
