import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp


def calc(grp, pixArea):

    med = grp.groupby(['healpixID', 'season'])['nights'].sum()

    nights_med = np.median(med)
    season_length_med = int(grp['season_length'].median())
    cadence_med = grp['cadence'].median()
    missing_nights = season_length_med/cadence_med-nights_med
    frac_missing_nights = 1.-(nights_med/(season_length_med/cadence_med))
    area = np.median(grp.groupby(['season']).size())*pixArea

    m5_single = {}
    for b in 'grizy':
        m5_single[b] = grp['m5_single_{}'.format(b)].median()

    return pd.DataFrame({'nights': [nights_med],
                         'season_length': [int(season_length_med)],
                         'cadence': [cadence_med],
                         'maxgap': [grp['maxgap'].median()],
                         'medgap': [grp['medgap'].median()],
                         'missing_nights': [missing_nights],
                         'frac_missing_nights': [frac_missing_nights],
                         'area': [area],
                         'm5_single_g':[m5_single['g']],
                         'm5_single_r':[m5_single['r']],
                         'm5_single_i':[m5_single['i']],
                         'm5_single_z':[m5_single['z']],
                         'm5_single_y':[m5_single['y']]})

def plot(ax, df, var):

    for dbName in np.unique(df['dbName']):
        idx = df['dbName'] == dbName
        sel = df[idx]
        #ax.plot(sel['fieldName'],sel[var], lineStyle='None',marker='o')
        ax.plot(sel['fieldName'], sel[var], marker='o', label=dbName)


def load(dbName, nside, group=['fieldName']):

    df = pd.DataFrame(np.load('DD_Summary_{}_{}.npy'.format(
        dbName, nside), allow_pickle=True))

    pixArea = hp.nside2pixarea(nside, degrees=True)
    
    for b in 'grizy':
        df['m5_single_{}'.format(b)] =  df['m5_{}'.format(b)]-1.25*np.log10(df['N_{}'.format(b)])
    
    print(df.columns)
    dfb = df.groupby(group).apply(lambda x: calc(x, pixArea)).reset_index()

    return dfb


def load_all(dbNames, nside, group=['fieldName']):

    df = pd.DataFrame()

    for dbName in dbNames:
        res = load(dbName, nside, group=group)
        res['dbName'] = dbName
        df = pd.concat((df, res))

    return df


dbNames = ['baseline_v1.5_10yrs',
           'descddf_v1.5_10yrs',
           'agnddf_v1.5_10yrs',
           'daily_ddf_v1.5_10yrs',
           # 'dm_heavy_nexp2_v1.6_10yrs',
           'dm_heavy_v1.6_10yrs',
           # 'ddf_heavy_nexp2_v1.6_10yrs',
           'ddf_heavy_v1.6_10yrs',
           'cadence_drive_gl200_gcbv1.6.1_10yrs']

nside = 128
dfb = load_all(dbNames, nside)

vars_toplot = ['nights', 'season_length', 'cadence',
               'maxgap', 'medgap', 'missing_nights', 'frac_missing_nights', 'area']
for b in 'grizy':
    vars_toplot += ['m5_single_{}'.format(b)]

legend = ['#nights/season/pixel', 'season length [day]', 'cadence [day-1]',
          'max gap [day]', 'median gap [day]', 'missing nights', 'frac missing_nights', 'area [deg2]']
for b in 'grizy':
    legend += ['m5 single visit - {} band'.format(b)]

toplot = dict(zip(vars_toplot, legend))


for key, val in toplot.items():
    fig, ax = plt.subplots()
    plot(ax, dfb, key)
    ax.set_ylabel(val)
    ax.legend()

plt.show()
