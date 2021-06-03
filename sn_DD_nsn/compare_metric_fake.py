import pandas as pd
from sn_tools.sn_io import loopStack
import glob
import matplotlib.pyplot as plt
from optparse import OptionParser

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

parser = OptionParser()

parser.add_option(
    '--metricDir', help='metric directory [%default]', default='../MetricOutput_DD_new_128',type=str)
parser.add_option(
    '--dbName', help='OS to process [%default]', default='daily_ddf_v1.5_10yrs', type=str)
parser.add_option(
    '--fakeDir', help='fake dir results [%default]', default='.', type=str)


opts, args = parser.parse_args()

metricDir = opts.metricDir
dbName = opts.dbName
fakeDir = opts.fakeDir

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
idx = df['nsnrat']>=2
pp = ['healpixID','season','zlim_faint','zlim']
print(df[idx][pp])
fig, ax = plt.subplots()

ax.plot(df['zlimdiff'], df['nsnrat'], 'ko')


plt.show()
