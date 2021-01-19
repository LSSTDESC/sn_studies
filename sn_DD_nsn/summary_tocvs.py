import numpy as np
import pandas as pd

prefix = 'DD_Summary'
nside = 128
dbName = 'descddf_v1.5_10yrs'
fieldName = 'COSMOS'

fName = '{}_{}_{}.npy'.format(prefix,dbName,nside)

tab = np.load(fName, allow_pickle=True)

idx = tab['fieldName'] == fieldName

sel = tab[idx]

print(sel.dtype)
x1= -2.0
color = 0.2
ebvofMW = 0.0
snrmin = 1.0
error_model = 1
bluecutoff = 380.
redcutoff = 800.
simulator = 'sn_fast'
fitter = 'sn_fast'

bands = 'grizy'
ro = []
for val in sel:
    r = ['{}_{}'.format(val['healpixID'],int(val['season']))]
    r += [x1,color,ebvofMW,snrmin,error_model,bluecutoff,redcutoff,simulator,fitter]
    for b in bands:
        r += [int(val['N_{}'.format(b)])]
    for b in bands:
        if val['N_{}'.format(b)] == 0:
            m5v = 21.
        else:
            m5v = val['m5_{}'.format(b)]-1.25*np.log10(val['N_{}'.format(b)])
        r += [m5v]
    for b in bands:
        r += [val['cadence']]
    r += [int(val['season'])]
    ro.append(r)

print(ro)
names=['tagprod','x1','color','ebvofMW','snrmin','error_model','bluecutoff','redcutoff','simulator','fitter','Ng','Nr','Ni','Nz','Ny','m5_g','m5_r','m5_i','m5_z','m5_y','cadence_g','cadence_r','cadence_i','cadence_z','cadence_y','season']
res = np.rec.fromrecords(ro, names=names)

print(res)
pd.DataFrame(res[:40]).to_csv('titooo.csv',index=None)
