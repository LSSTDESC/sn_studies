import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd


def plot_Single(dirFiles, theFile, cadence):

    bands = 'grizy'

    colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
    tab = np.load('{}/{}'.format(dirFiles, theFile), allow_pickle=True)

    idx = np.abs(tab['cadence']-cadence) < 1.e-5
    sel = tab[idx]

    sel.sort(order='z')
    fig, ax = plt.subplots()
    fig.suptitle('{}'.format(sel['min_par']))
    ax.plot(sel['z'], sel['Nvisits'], color='k')

    for b in bands:
        ax.plot(sel['z'], sel['Nvisits_{}'.format(b)],
                color=colors[b], label='{} band'.format(b))

    ax.grid()
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('Nvisits', fontsize=12)
    plt.legend()


def plot_Multiple(dirFiles, files, cadence):

    fig, ax = plt.subplots()

    for index, row in files.iterrows():
        fName = '{}/{}'.format(dirFiles, row['file'])
        tab = np.load(fName, allow_pickle=True)
        print(tab.dtype)
        idx = np.abs(tab['cadence']-cadence) < 1.e-5
        sel = tab[idx]
        sel.sort(order='z')
        ax.plot(sel['z'], sel['Nvisits'], label='{}'.format(
            np.unique(sel['min_par']).item()))

    ax.grid()
    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel('Nvisits', fontsize=12)
    plt.legend()


parser = OptionParser()

parser.add_option('--dirFiles', type=str, default='dd_design_faint/Nvisits_z',
                  help='dir location of the file [%default]')
parser.add_option('--fileList', type=str, default='Nvisits_z.csv',
                  help='list of files to process [%default]')
parser.add_option('--cadence', type=int, default=3,
                  help='cadence of observation [%default]')
parser.add_option('--displayType', type=str, default='Nvisits_all',
                  help='type of display: Nvisits_all/Nvisits_band [%default]')

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
fileList = opts.fileList
cadence = opts.cadence
displayType = opts.displayType

files = pd.read_csv(fileList, comment='#')

if displayType == 'Nvisits_all':
    plot_Multiple(dirFiles, files, cadence)
if displayType == 'Nvisits_band':
    for index, row in files.iterrows():
        plot_Single(dirFiles, row['file'], cadence)

plt.show()
