import numpy as np
from sn_saturation import plt, filtercolors

dbDir = '../DB_Files'
dbName = 'baseline_nexp1_v1.7_10yrs.npy'

fullName = '{}/{}'.format(dbDir, dbName)

data = np.load(fullName)

fig, ax = plt.subplots(figsize=(10, 8))

bands = 'gri'
ls = dict(zip(bands, ['solid', 'dashed', 'dotted']))
filtcols = dict(zip(bands, 'bgr'))
var = 'seeingFwhmEff'

for b in bands:
    idx = data['band'] == b
    sel = data[idx]
    med = np.round(np.median(sel[var]), 2)
    ll = '{} band (median: {}\'\')'.format(b, med)
    ax.hist(sel[var], histtype='step',
            bins=100, density=True, lw=2, stacked=True, color=filtcols[b], ls=ls[b], label=ll)

ax.set_xlim([0.5, 2.5])
ax.set_xlabel('seeing [\'\']')
ax.set_ylabel('Number of Entries')
ax.legend(frameon=False)
plt.show()
