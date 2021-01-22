import numpy as np
import pandas as pd
from optparse import OptionParser
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def all_nights(dbDir,dbName,dbExtens):
    
    fullName = '{}/{}.{}'.format(dbDir,dbName,dbExtens)

    df = pd.DataFrame(np.load(fullName, allow_pickle=True))

    # get moon* columns
    r = []
    for bb in df.columns:
        if 'moon' in bb:
            r.append(bb)

    # get medians here

    meds = df.groupby(['night'])[r].median().reset_index()

    print(meds)

    # now estimate all the nights
    night_min = meds['night'].min()
    night_max = meds['night'].max()

    nights = pd.DataFrame(range(night_min,night_max+1),columns=['night'])
    nights['downtime'] = 0
    idx = nights['night'].isin(meds['night'])
    nights.loc[~idx,'downtime'] = 1

    interp = interpol(r,meds)

    # now add moon values to night_df(from interp)
    for vv in r:
        nights[vv] = interp[vv](nights['night'])
    
    return nights,interp

def interpol(r,meds):

    interp = {}
    for vv in r:
        interp[vv]  = interp1d(meds['night'], meds[vv], bounds_error=False, fill_value=0.)

    return interp

def load_DD(fieldDir,nside,dbName,fieldName):
    
    fullName = '{}/ObsPixelized_{}_{}_{}_night.npy'.format(fieldDir,nside,dbName,fieldName)
    tab = np.load(fullName,allow_pickle=True)

    return np.copy(tab)

def filterseq(grp):

    seq = ''.join(grp['filter'].tolist())

    return pd.DataFrame({'filterseq': [seq]})


def selectSeason(DD_field, nights, season=1):

    idx = DD_field['season'] == season
    sel_DD = DD_field[idx]
    night_min = sel_DD['night'].min()
    night_max = sel_DD['night'].max()

    ido = nights['night'] >= night_min
    ido &= nights['night'] <= night_max
    nights_sel = nights[ido]
    
    return nights_sel,sel_DD
    
parser = OptionParser()

parser.add_option("--dbDir", type=str, default='../DB_Files',
                  help="OS dir location[%default]")
parser.add_option("--dbName", type=str, default='descddf_v1.5_10yrs',
                  help="OS name[%default]")
parser.add_option("--dbExtens", type=str, default='npy',
                  help="OS extens (db or npy) [%default]")
parser.add_option("--fieldName", type=str, default='COSMOS',
                  help="field to consider for this study  [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="healpix nside [%default]")
parser.add_option("--fieldDir", type=str, default='.',
                  help="dir where the field file is  [%default]")

opts, args = parser.parse_args()

# get infos for all the nights

nights, interp = all_nights(opts.dbDir,opts.dbName,opts.dbExtens)

DD_field = pd.DataFrame(load_DD(opts.fieldDir,opts.nside,opts.dbName,opts.fieldName))
#add vars here

for key, vals in interp.items():
    DD_field[key] = vals(DD_field['night'])


fig, ax = plt.subplots()

nights_sel,DD_sel = selectSeason(DD_field,nights,2)

idx = nights_sel['downtime'] == 0
nights_on = nights_sel[idx]
nights_off = nights_sel[~idx]

what = 'moonPhase'
ax.plot(nights_on['night'],nights_on[what],'ko',mfc='None')
ax.plot(nights_off['night'],nights_off[what],'ro',mfc='None')
DD_seq = DD_sel.groupby(['healpixID','night']).apply(lambda x: filterseq(x)).reset_index()
rrec = DD_seq.to_records(index=False)
rrec = pd.DataFrame(np.unique(rrec[['night','filterseq']]))
for key, vals in interp.items():
    rrec[key] = vals(rrec['night'])

#idx = np.where((rrec['filterseq'] == 'gri') or (rrec['filterseq'] == 'zy'))
idx = rrec['filterseq'] == 'gri'
print(len(rrec[idx]),len(rrec))
ax.plot(rrec[idx]['night'],rrec[idx][what],'b*')
idx = rrec['filterseq'] == 'zy'
ax.plot(rrec[idx]['night'],rrec[idx][what],'m*')

plt.show()
