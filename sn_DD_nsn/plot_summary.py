import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(data,whatx='zlim', whaty='nsn'):

    fig, ax = plt.subplots()

    for dbName in dbNames:
        idx = data['dbName'] == dbName
        df_sel = data.loc[idx]
        if 'fieldName' in data.columns:
            for fieldName in fieldNames:    
                idxb = df_sel['fieldName'] == fieldName
                df_selb = df_sel.loc[idx]
                ax.plot(df_selb[whatx],df_selb[whaty], color=colors[dbName],marker='.',lineStyle='None')
        else:
            ax.plot(df_sel[whatx],df_sel[whaty], color=colors[dbName],marker='.',lineStyle='None')

    ax.set_xlabel(whatx)
    ax.set_ylabel(whaty)
fis = glob.glob('nSN_zlim_DD_*')

df_data = pd.DataFrame()
for fi in fis:
    tab = np.load(fi, allow_pickle=True)
    print(tab.dtype)
    df_data = pd.concat((df_data, pd.DataFrame(tab)))



    
fieldNames = list(df_data['fieldName'].unique())
dbNames = list(df_data['dbName'].unique())
color_ref = ['b','g','r','m','c',[0.8,0,0]]

colors = dict(zip(dbNames, color_ref[:len(dbNames)]))
print('hello',fieldNames, dbNames)
groupby = ['dbName']
#groupby += ['fieldName']
#groupby += ['healpixID']

df_data_group = df_data.groupby(groupby).apply(lambda x: pd.DataFrame({'nsn': [x['nsn'].sum()],
                                                                       'zlim': [x['zlim'].median()],
                                                                       'cadence' : [x['cadence'].median()],
                                                                       'season_length' : [x['season_length'].median()]})).reset_index()

print(df_data_group)

plot(df_data_group)
plot(df_data_group,'season_length','nsn')
plot(df_data_group,'cadence','nsn')

plt.show()                                              

                                                                            
