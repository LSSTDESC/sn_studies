import pandas as pd
from sn_tools.sn_io import loopStack
import glob
import matplotlib.pyplot as plt


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


metricDir = '../../MetricOutput_DD_new_128'
dbName = 'daily_ddf_v1.5_10yrs'
fakeDir = '.'

df_metric = loadMetric(metricDir, dbName)
df_fakes = loadFakes(fakeDir, dbName)

print(df_metric)

print(df_fakes)

df = df_fakes.merge(df_metric, left_on=['healpixID', 'season'], right_on=[
                    'healpixID', 'season'])

df['zlimdiff'] = df['zlim_faint']-df['zlim']
df['nsnrat'] = df['nsn_med_faint']/df['nsn_exp']
print(df[['zlimdiff', 'nsnrat']])
print(df.columns)
fig, ax = plt.subplots()

ax.plot(df['gap_max'], df['nsnrat'], 'ko')


plt.show()
