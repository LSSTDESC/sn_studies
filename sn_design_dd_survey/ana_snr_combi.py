import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import glob
from optparse import OptionParser
import multiprocessing
import operator


def min_nvisits(z, snr, colout, mincol='Nvisits', minpar='nvisits', select={}):
    """
    Method the combi with the lower number of visits

    Parameters
    ---------------
    snr: pandas df
    data to process
    mincol: str
    the colname where the min has to apply (default: Nvisits)
    select: dict
    selection (contrain) to apply (default: {})

    Returns
    -----------
    the ten first rows with the lower nvisits

    """

    if select:
        if z >= select['zmin']:
            idx = True
            for key, vals in select.items():
                if key != 'zmin':
                    idx &= vals['op'](snr[vals['var']], vals['value'])
                    snr = snr[idx]
    if mincol != 'Nvisits':
        snr = snr.sort_values(by=['Nvisits', mincol])
    else:
        snr = snr.sort_values(by=[mincol])
    snr['min_par'] = minpar
    snr['min_val'] = snr[mincol]

    nout = np.min([len(snr), 10])
    return snr[colout][:nout]


def load_multiple(thedir, snrfi, nproc=8):
    """
    Method to load and concatenate a set of npy files

    Parameters
    ---------------
    thedir: str
       location directory of the file
    fi: str
       name of the files (without npy extension)
    nproc: int, opt
      number of procs to use (default: 8)

    Returns
    -----------
    tab: pandas df
      data

    """
    fi = glob.glob('{}/{}_*.npy'.format(thedir, snrfi))
    nfis = len(fi)
    batch = np.linspace(0, nfis, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    for i in range(nproc):

        ida = batch[i]
        idb = batch[i+1]

        p = multiprocessing.Process(
            name='Subprocess', target=load, args=(fi[ida:idb], i, result_queue))
        p.start()

    resultdict = {}

    for j in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    df = pd.DataFrame()
    for j in range(nproc):
        df = pd.concat((df, resultdict[j]))

    return df


def load(fi, j=0, output_q=None):
    """
    Method to load and concatenate a set of npy files

    Parameters
    ---------------
    fi: list
       list of files to process
    j: int, opt
       multiproc number (default: 0)
    output_q: multiprocessing queue

    Returns
    -----------
    tab: pandas df
      data

    """

    snrtot = None
    for ff in fi:
        rr = np.load(ff, allow_pickle=True)
        if snrtot is None:
            snrtot = rr
        else:
            snrtot = np.concatenate((snrtot, rr))

    tab = pd.DataFrame(snrtot)

    if output_q is not None:
        output_q.put({j: tab})
    else:
        return tab


def plotb(tab, z, whata='Nvisits', whatb='Nvisits', leg='$N_{visits}$',
          bands='gr', colors=dict(zip('grizy', 'bgrym'))):
    """
    Method to plot results of SNR combis

    Parameters
    ---------------
    tab: pandas df
      data to plot
    whata: str, opt
      x axis variable to plot (default: Nvisits)
     whaty: str, opt
      y axis variable to plot (default: Nvisits)
    leg: str, opt
      x axis legend (default: ='$N_{visits}$'
    bands: str, opt
      bands to display (default: gr)
    colors: dict, opt
      dict of the color per band (default: dict(zip('grizy', 'bgrym')))
    """

    fontsize = 15
    fig, ax = plt.subplots()
    cadence = '1 day$^{-1}$'
    sigmaC = '$\sigma_{color} \sim 0.04\pm1\%$'
    fig.suptitle('($x_1,color$)=({},{}) - z={} \n cadence={} - {}'.format(-2.0,
                                                                          0.2, z, cadence, sigmaC), fontsize=fontsize)

    for b in bands:
        sel = tab
        if 'SNR' in whata:
            idx = tab[whata] < 1000.
            sel = tab[idx]

        if 'SNR' in whatb:
            idx = tab[whatb] < 1000.
            sel = tab[idx]

        ax.plot(sel['{}_{}'.format(whata, b)],
                sel['{}_{}'.format(whatb, b)], '{}.'.format(colors[b]), label='${}$-band'.format(b))

    ax.set_xlabel(whata, fontsize=fontsize)
    ax.set_ylabel(whatb, fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
    ax.grid()
    ax.legend(fontsize=fontsize-3)
    # minx = int(np.min(tab['Nvisits']))
    # ax.set_xlim(minx-1, 200)
    # ax.set_ylim(0, 200)
    """
    ll = [minx]
    ll += range(60, 220, 20)
    ax.set_xticks(ll)
    """


def plot(tab, z, whata='Nvisits', whatb='Nvisits', legx='$N_{visits}$', legy='$N_{visits}^{band}$', bands='gr', colors=dict(zip('grizy', 'bgrym'))):
    """
    Method to plot results of SNR combis

    Parameters
    ---------------
    tab: pandas df
      data to plot
    whata: str, opt
      x axis variable to plot (default: Nvisits)
     whaty: str, opt
      y axis variable to plot (default: Nvisits)
    legx: str, opt
      x axis legend (default: ='$N_{visits}$'
     legy: str, opt
      y axis legend (default: ='$N_{visits}^{bands}$'
    bands: str, opt
      bands to display (default: gr)
    colors: dict, opt
      dict of the color per band (default: dict(zip('grizy', 'bgrym')))

    """

    fontsize = 15
    fig, ax = plt.subplots()
    cadence = '1 day$^{-1}$'
    sigmaC = '$\sigma_{color} \sim 0.04\pm1\%$'
    fig.suptitle('($x_1,color$)=({},{}) - z={} \n cadence={} - {}'.format(-2.0,
                                                                          0.2, z, cadence, sigmaC), fontsize=fontsize)

    for b in bands:
        sel = tab
        if 'SNR' in whata:
            idx = tab[whata] < 1000.
            sel = tab[idx]

        if 'SNR' in whatb:
            idx = tab[whatb] < 1000.
            sel = tab[idx]
        ax.plot(sel[whata], sel['{}_{}'.format(whatb, b)],
                '{}.'.format(colors[b]), label='${}$-band'.format(b))

    ax.set_xlabel(legx, fontsize=fontsize)
    ax.set_ylabel(legy, fontsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
    ax.grid()
    ax.legend(fontsize=fontsize-3)
    """
    minx = int(np.min(tab['Nvisits']))
    ax.set_xlim(minx-1, 200)
    ax.set_ylim(0, 200)
    ll = [minx]
    ll += range(60, 220, 20)
    ax.set_xticks(ll)
    """


parser = OptionParser()

parser.add_option("--dirFiles", type="str", default='SNR_combi',
                  help="location dir of files [%default]")
parser.add_option("--z", type=float, default=0.7,
                  help="redshift value for display[%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="SN stretch[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color[%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of procs to use[%default]")


opts, args = parser.parse_args()

z = np.round(opts.z, 2)
dirFiles = '{}/z_{}'.format(opts.dirFiles, z)

x1 = np.round(opts.x1, 1)
color = np.round(opts.color, 1)
snrfi = 'SNR_combi_{}_{}_{}'.format(x1, color, z)

tab = load_multiple(dirFiles, snrfi, opts.nproc)


print('hello', tab.filter(regex='sigma').columns)

tab = tab.sort_values(by=['Nvisits'])
idx = tab['Nvisits'] < 100000000.
#idx &= tab['sigmaC'] >= 0.0390
snr = tab[idx]
"""
sel = sel.rename(columns={'SNRcalc_tot': 'SNRcalc'})
print(sel.columns)
colors = dict(zip('grizy', 'bgrym'))
bands = 'r'
plot(sel, z, bands=bands, colors=colors)

# plot(sel, whata='Nvisits', whatb='SNRcalc', legy='$SNR_{band}$')

plot(sel, z, whata='sigmaC', whatb='SNRcalc',
     legx='sigmaC', legy='$SNR_{band}$', bands=bands, colors=colors)
plotb(sel, z, 'SNRcalc', 'Nvisits', bands=bands, colors=colors)


plt.show()
"""
colout = ['sigmaC', 'SNRcalc_tot', 'Nvisits', 'Nvisits_g', 'SNRcalc_g',
          'Nvisits_r', 'SNRcalc_r', 'Nvisits_i', 'SNRcalc_i', 'Nvisits_z',
                       'SNRcalc_z', 'Nvisits_y', 'SNRcalc_y', 'min_par', 'min_val']

colout = ['sigmaC', 'Nvisits', 'Nvisits_g', 'Nvisits_r', 'Nvisits_i', 'Nvisits_z',
          'Nvisits_y', 'min_par', 'min_val']

colout = ['sigmaC', 'Nvisits',
          'Nvisits_r', 'SNRcalc_r', 'Nvisits_i', 'SNRcalc_i', 'Nvisits_z',
                       'SNRcalc_z', 'Nvisits_y', 'SNRcalc_y']

sel = snr.copy()

sel['Delta_iz'] = np.abs(sel['Nvisits_i']-sel['Nvisits_z'])
sel['Delta_SNR'] = sel['SNRcalc_z']-sel['SNRcalc_i']

seldict = {}
seldict['zmin'] = 0.6
seldict['cut1'] = {}
seldict['cut1']['var'] = 'SNRcalc_r'
seldict['cut1']['value'] = 5.
seldict['cut1']['op'] = operator.le
seldict['cut2'] = {}
seldict['cut2']['var'] = 'SNRcalc_g'
seldict['cut2']['value'] = 5.
seldict['cut2']['op'] = operator.le

seldictb = seldict.copy()
seldictb['cut3'] = {}
seldictb['cut3']['var'] = 'Delta_SNR'
seldictb['cut3']['value'] = 0.
seldictb['cut3']['op'] = operator.ge

selvar = ['Nvisits', 'Nvisits_y', 'Delta_iz']
minparname = ['nvisits', 'nvisits_y', 'deltav_iz']
combi = dict(zip(selvar, minparname))
snr_visits = pd.DataFrame()

for key, val in combi.items():
    res = min_nvisits(z, sel, colout, key, val, seldict)
    print('parameter', key)
    print(res)
