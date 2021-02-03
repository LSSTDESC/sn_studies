import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--dirFiles', type=str, default='dd_design_faint/Nvisits_z',
                  help='dir location of the file [%default]')
parser.add_option('--theFile', type=str, default='Nvisits_z_-2.0_0.2_error_model_ebvofMW_0.0_nvisits.npy',
                  help='file to process [%default]')
parser.add_option('--cadence', type=int, default=3,
                  help='cadence of observation [%default]')

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
theFile = opts.theFile
cadence = opts.cadence


bands = 'grizy'

colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
tab = np.load('{}/{}'.format(dirFiles, theFile), allow_pickle=True)

idx = np.abs(tab['cadence']-cadence) < 1.e-5
sel = tab[idx]

sel.sort(order='z')
fig, ax = plt.subplots()

ax.plot(sel['z'], sel['Nvisits'], color='k')

for b in bands:
    ax.plot(sel['z'], sel['Nvisits_{}'.format(b)],
            color=colors[b], label='{} band'.format(b))

ax.grid()
ax.set_xlabel('z', fontsize=12)
ax.set_ylabel('Nvisits', fontsize=12)
plt.legend()
plt.show()
