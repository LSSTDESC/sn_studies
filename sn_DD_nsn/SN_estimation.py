import glob
from sn_tools.sn_io import loopStack
from astropy.table import Table, vstack
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def finalNums(grp,normfact=10.):

    return pd.DataFrame({'nsn_zlim': [grp['nsn_zlim'].sum()/normfact],
                         'nsn_tot': [grp['nsn_tot'].sum()/normfact],
                         'zlim': [grp['zlim'].median()]})

def nSN(grp,sigmaC=0.04):
    
    idx = grp['Cov_colorcolor']<=sigmaC**2
    sela = grp[idx]
    idx &= grp['z']<= grp['zlim']
    sel = grp[idx]

    return pd.DataFrame({'nsn_zlim':[len(sel)],
                         'nsn_tot':[len(sela)],
                         'zlim': [grp['zlim'].median()]})

def SN(fitDir,dbName,fieldNames,SNtype):

    dfSN = pd.DataFrame()
    for field in fieldNames:
        fis = glob.glob('{}/{}/*{}*{}*.hdf5'.format(fitDir,dbName,field,SNtype))
        
        out = loopStack(fis,objtype='astropyTable').to_pandas()
        out['fieldName'] = field
        idx = out['Cov_colorcolor']>=1.e-5
        dfSN = pd.concat((dfSN, out[idx]))

    return dfSN

def zlim(grp,sigmaC=0.04):
    
    ic = grp['Cov_colorcolor']<=sigmaC**2
    selb = grp[ic]

    if len(selb)==0:
        zl = 0.
    else:
        zl = np.max(selb['z'])

    return pd.DataFrame({'zlim':[zl]})


mainDir = '/sps/lsst/users/gris/DD'
fitDir = '{}/Fit'.format(mainDir)
simuDir = '{}/Simu'.format(mainDir)

dbName = 'descddf_v1.5_10yrs'
fieldNames = ['COSMOS','CDFS','ELAIS','XMM-LSS','ADFS1','ADFS2']

allSN = pd.DataFrame()
zlimit = None

#get faintSN
                        
faintSN = SN(fitDir,dbName,fieldNames,'faintSN')
allSN = SN(fitDir,dbName,fieldNames,'allSN')

zlimit = faintSN.groupby(['healpixID','fieldName','season']).apply(lambda x:zlim(x))

print(zlimit)
allSN=allSN.merge(zlimit,left_on=['fieldName', 'season'],right_on=['fieldName','season'])

sumSN = allSN.groupby(['healpixID','fieldName','season']).apply(lambda x: nSN(x)).reset_index()

print(sumSN.groupby(['fieldName']).apply(lambda x: finalNums(x)).reset_index())

print(allSN.columns)


"""
tab = out['fullSN']

for season in np.unique(tab['season']):
    idx = tab['season'] == season
    sel = tab[idx]
    ib = zlimit
plt.plot(tab['z'],np.sqrt(tab['Cov_colorcolor']),'ko')
plt.show()
"""
