import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
"""
def nstat(grp):

    dd = {}
    for vv in ['gir','yz','giryz','g','giruy','r','i','z','y']:
        dd[vv] = [grp['mean_nights_{}'.format(vv)].sum()
"""
def calc(grp):

    print(grp.name)
    
    dictres = {}
    dictres['mean_nonights'] = [np.mean(grp['nights_off']/grp['nights_all'])]
    dictres['std_nonights'] = [np.std(grp['nights_off']/grp['nights_all'])]
    dictres['mean_cadence'] = [np.mean(grp['cadence'])]
    dictres['std_cadence'] = [np.std(grp['cadence'])]
    dictres['mean_season_length'] = [np.mean(grp['nights_all'])]
    dictres['std_season_length'] = [np.std(grp['nights_all'])]
    for vv in ['gir','yz','giryz','g','giruy','r','i','z','y']:
        dictres['mean_nights_{}'.format(vv)] =[np.mean(grp['nights_{}'.format(vv)]/grp['nights_ddf'])]
        dictres['rms_nights_{}'.format(vv)] =[np.std(grp['nights_{}'.format(vv)]/grp['nights_ddf'])]

    main_seq = 'giryz'
    factor_seq = 1
    if 'descddf' in grp.name[1]:
        main_seq = 'yz'
        factor_seq = 2
    dictres['mean_nights_seq'] = [factor_seq*dictres['mean_nights_{}'.format(main_seq)][0]]
    
    return pd.DataFrame.from_dict(dictres)

def load(fis):
    dftot = pd.DataFrame()
    for fi in fis:
        dbName = fi.split('/')[-1]
        dbName = '_'.join(dbName.split('_')[:-2])
        df = pd.read_csv(fi, comment='#')
        print(dbName, df)
        #fig, ax = plt.subplots()
        #ax.plot(df['fieldName'],df['nights_off']/df['nights_all'],'ko')
        df['nights_ddf'] = 0
        for vv in ['gir','yz','giryz','g','giruy','r','i','z','y']:
            df['nights_ddf'] += df['nights_{}'.format(vv)]
        db = df.groupby(['fieldName','dbName']).apply(lambda x: calc(x)).reset_index()
        db['dbName_spl'] = dbName
        
        dftot = pd.concat((dftot,db))
    return dftot

def plot(dftot, xvar='dbName_spl',yvar='mean_nonights',yerr='std_nonights',legy='Fraction of nights without observation'):

    #fig, ax = plt.subplots(nrows=2)
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True,figsize=(12,7))

    plt.subplots_adjust(hspace=.0)
    io=-1
    for fieldName in dftot['fieldName'].unique():
        io += 1
        idx = dftot['fieldName'] == fieldName
        sel = dftot[idx]
        sel = sel.sort_values(by=xvar)
        #ax.errorbar(sel[xvar],sel[yvar],yerr=sel[yerr],capsize=2.,label=fieldName)
        ax1.plot(sel[xvar],sel[yvar],label=fieldName)
        ax2.plot(sel[xvar],sel[yerr],label=fieldName)
        #print(sel[['dbName','mean','std']])

    fontsize = 12
    ax1.set_ylabel('mean {}'.format(legy),fontsize=fontsize)
    ax1.tick_params(axis='x', labelrotation=20.)
    ax2.set_ylabel('RMS {}'.format(legy),fontsize=fontsize)
    ax2.tick_params(axis='x', labelrotation=20.)

    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax1.grid()
    ax2.grid()
    plt.legend(bbox_to_anchor=(0.95,2.2),ncol=6,fontsize=fontsize,frameon=False)

def plot_new(dftot, xvar='dbName_spl',yvar='mean_nights_seq',legy='Fraction of nights \n with the full sequence'):

    #fig, ax = plt.subplots(nrows=2)
    fig, ax = plt.subplots(figsize=(12,7))

    io=-1
    for fieldName in dftot['fieldName'].unique():
        io += 1
        idx = dftot['fieldName'] == fieldName
        sel = dftot[idx]
        sel = sel.sort_values(by=xvar)
        #ax.errorbar(sel[xvar],sel[yvar],yerr=sel[yerr],capsize=2.,label=fieldName)
        ax.plot(sel[xvar],sel[yvar],label=fieldName)
        #print(sel[['dbName','mean','std']])

    fontsize = 12
    ax.set_ylabel('mean {}'.format(legy),fontsize=fontsize)
    ax.tick_params(axis='x', labelrotation=20.)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.grid()

    plt.legend(bbox_to_anchor=(0.95,1.1),ncol=6,fontsize=fontsize,frameon=False)

    
theDir = '../OS_downtime_old'

fis = glob.glob('{}/*.csv'.format(theDir))

df = load(fis)

print(df.columns)
plot(df,yvar='mean_nonights',yerr='std_nonights',legy='fraction of nights \n without observation')
plot(df, yvar='mean_cadence',yerr='std_cadence',legy='cadence \n of observation [days$^{-1}$]')
plot(df, yvar='mean_season_length',yerr='std_season_length',legy='season length [days]')

plot_new(df, yvar='mean_nights_seq')
plot_new(df, yvar='mean_nights_giruy',legy='griuy')
plot_new(df, yvar='mean_nights_y',legy='y')
plot_new(df, yvar='mean_nights_gir',legy='gir')
df['nn'] = 0
for vv in ['gir','yz','giryz','g','giruy','r','i','z','y']:
    what = 'mean_nights_{}'.format(vv)
    idx = df[what]>0.
    if len(df[idx])>0:
        print('yy',vv)
    df['nn'] += df[what]
    
print(df['nn'])
plt.show()
