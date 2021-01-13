import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc(grp):

    med = grp.groupby(['healpixID','season'])['nights'].sum()

    return pd.DataFrame({'nights': [np.mean(med)],
                         'season_length': [int(grp['season_length'].median())], 
                         'cadence': [grp['cadence'].median()], 
                         'maxgap' : [grp['maxgap'].median()], 
                         'medgap': [grp['medgap'].median()]})

def plot(ax, df, var):


    for dbName in np.unique(df['dbName']):
        idx = df['dbName']==dbName
        sel = df[idx]
        #ax.plot(sel['fieldName'],sel[var], lineStyle='None',marker='o')
        ax.plot(sel['fieldName'],sel[var], marker='o',label=dbName)

def load(dbName, group=['fieldName']):
    
    df = pd.DataFrame(np.load('DD_Summary_{}.npy'.format(dbName),allow_pickle=True))


    dfb = df.groupby(group).apply(lambda x : calc(x)).reset_index()

    return dfb

def load_all(dbNames, group=['fieldName']):

    df = pd.DataFrame()
    
    for dbName in dbNames:
        res = load(dbName, group=group)
        res['dbName'] = dbName
        df = pd.concat((df,res))

    return df

dbNames=['descddf_v1.5_10yrs','agnddf_v1.5_10yrs','baseline_v1.5_10yrs','daily_ddf_v1.5_10yrs']

dfb = load_all(dbNames)

vars_toplot = ['nights','season_length','cadence','maxgap','medgap']
legend = ['#nights','season length [day]','cadence [day-1]','max gap [day]','median gap [day]']

toplot=dict(zip(vars_toplot,legend))


for key, val in toplot.items():
    fig, ax = plt.subplots()
    plot(ax, dfb, key)
    ax.set_ylabel(val)
    ax.legend()

plt.show()
