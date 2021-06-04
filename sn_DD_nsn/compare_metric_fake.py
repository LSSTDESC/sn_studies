import pandas as pd
from sn_tools.sn_io import loopStack
import glob
import matplotlib.pyplot as plt
from optparse import OptionParser
import scipy.stats
import numpy as np


def loadMetric(metricDir, dbName):

    metricfis = glob.glob('{}/{}/*/*.hdf5'.format(metricDir, dbName))

    metric = loopStack(metricfis, 'astropyTable')

    return metric.to_pandas()


def loadFakes(fakeDir, dbName):

    fakes_cv = glob.glob('config_DD*zlim.csv')

    df = pd.DataFrame()

    for ff in fakes_cv:
        do = pd.read_csv(ff)
        df = pd.concat((df, do))

    return df


def plotBinned(ax, metricTot, xp='cadence', yp='nsn_med_faint', label='', bins=10, therange=(1., 10.)):

    x = metricTot[xp]
    y = metricTot[yp]

    means_result = scipy.stats.binned_statistic(
        x, [y, y**2], bins=10, range=therange, statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    ax.errorbar(x=bin_centers, y=means, yerr=standard_deviations,
                marker='.', label=label)


parser = OptionParser()

parser.add_option(
    '--metricDir', help='metric directory [%default]', default='../MetricOutput_DD_new_128', type=str)
parser.add_option(
    '--dbName', help='OS to process [%default]', default='daily_ddf_v1.5_10yrs', type=str)
parser.add_option(
    '--fakeDir', help='fake dir results [%default]', default='.', type=str)


opts, args = parser.parse_args()

metricDir = opts.metricDir
dbName = opts.dbName
fakeDir = opts.fakeDir

df_ = loadMetric(metricDir, dbName)
idx = df_['nsn_med_faint'] >= 0
df_metric = df_[idx]
df_metric['nsn_med_faint_norm'] = 180. * \
    df_metric['nsn_med_faint'] / df_metric['season_length']
print(df_metric.columns)
healpixID = 144428

idx = df_metric['healpixID'] == healpixID
vars = ['healpixID', 'season', 'cadence',
        'zlim_faint', 'nsn_med_faint', 'nsn_med_faint_norm', 'season_length', 'gap_max', 'gap_med']
print(df_metric[idx][vars])

plt.plot(df_metric['gap_max'], df_metric['cadence'], 'ko')

fig, ax = plt.subplots()

plotBinned(ax, df_metric, xp='gap_max', yp='cadence',
           label='', bins=10, therange=(1., 50.))
plt.show()
"""
df_fakes = loadFakes(fakeDir, dbName)

print(df_metric)

print(df_fakes)

df = df_fakes.merge(df_metric, left_on=['healpixID', 'season'], right_on=[
                    'healpixID', 'season'])

df['zlimdiff'] = df['zlim_faint']-df['zlim']
df['nsnrat'] = df['nsn_med_faint']/df['nsn_exp']
print(df[['zlimdiff', 'nsnrat']])
print(df.columns)
idx = df['nsnrat']>=2
pp = ['healpixID','season','zlim_faint','zlim']
print(df[idx][pp])
fig, ax = plt.subplots()

ax.plot(df['zlimdiff'], df['nsnrat'], 'ko')

plt.show()
"""
