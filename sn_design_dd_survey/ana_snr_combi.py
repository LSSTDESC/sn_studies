import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import glob
from optparse import OptionParser
import multiprocessing
import operator


def save_file(tab):
    """
    Function to save in a cvs file a list of combi infos

    Parameters
    ---------------
    tab: pandas df
       data to process

    """
    res = pd.DataFrame(tab)
    print(tab.columns)

    res['x1'] = -2.0
    res['color'] = 0.2
    res['ebvofMW'] = 0.0
    res['snrmin'] = 1.
    res['error_model'] = 1
    res['errmodrel'] = 0.1
    res['bluecutoff'] = 380.
    res['redcutoff'] = 800.
    res['simulator'] = 'sn_fast'
    res['fitter'] = 'sn_cosmo'
    res['season'] = 1
    res['tagprod'] = np.arange(res.shape[0])
    for b in 'grizy':
        res['cad_{}'.format(b)] = 1

    name_in = []
    name_out = []
    round_val = []
    forint = []
    for b in 'grizy':
        name_in.append('Nvisits_{}'.format(b))
        name_out.append('N{}'.format(b))
        round_val.append(0)
        res['Nvisits_{}'.format(b)] = res['Nvisits_{}'.format(b)].astype(int)

    for b in 'grizy':
        name_in.append('m5single_{}'.format(b))
        name_out.append('m5_{}'.format(b))
        round_val.append(2)

    for b in 'grizy':
        name_in.append('cad_{}'.format(b))
        name_out.append('cadence_{}'.format(b))
        round_val.append(0)

    rename = dict(zip(name_in, name_out))
    round_all = dict(zip(name_in, round_val))
    print('hhh', rename)
    res = res.round(round_all)
    res = res.rename(columns=rename)
    print(res.columns)
    what_to_dump = ['tagprod', 'x1', 'color', 'ebvofMW', 'snrmin', 'error_model',
                    'errmodrel', 'bluecutoff', 'redcutoff', 'simulator', 'fitter']

    what_to_dump += name_out
    what_to_dump += ['season']

    outName = 'toto.csv'
    res[what_to_dump].to_csv(outName, index=False)


def min_nvisits(z, snr, colout, mincol='Nvisits', minpar='nvisits', select={}):
    """
    Function  the combi with the lower number of visits

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
    print('hello', len(snr), mincol)
    if mincol != 'Nvisits':
        snr = snr.sort_values(by=['Nvisits', mincol])
    else:
        snr = snr.sort_values(by=[mincol])
    snr['min_par'] = minpar
    snr['min_val'] = snr[mincol]

    nout = np.min([len(snr), 10])
    return snr[colout]


def load_multiple(thedir, snrfi, sigmaC_min, sigmaC_max, nproc=8):
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
    print('loading', fi)
    batch = np.linspace(0, nfis, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    for i in range(nproc):

        ida = batch[i]
        idb = batch[i+1]

        p = multiprocessing.Process(
            name='Subprocess', target=load, args=(fi[ida:idb], sigmaC_min, sigmaC_max, i, result_queue))
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


def load(fi, sigmaC_min=0.99*0.04, sigmaC_max=1.01*0.04, j=0, output_q=None):
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
        # print(rr.dtype)
        idx = rr['sigmaC'] >= sigmaC_min
        idx &= rr['sigmaC'] <= sigmaC_max

        #idx &= rr['Nvisits'] <= 80.
        sel = np.copy(rr[idx])
        if len(sel) > 0:
            if snrtot is None:
                snrtot = sel
            else:
                snrtot = np.concatenate((snrtot, sel))

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
parser.add_option("--sigmaC_min", type=float, default=0.99*0.04,
                  help="sigmaC min value[%default]")
parser.add_option("--sigmaC_max", type=float, default=1.01*0.04,
                  help="sigmaC max value[%default]")


opts, args = parser.parse_args()

z = np.round(opts.z, 2)
dirFiles = '{}/z_{}'.format(opts.dirFiles, z)

x1 = np.round(opts.x1, 1)
color = np.round(opts.color, 1)
snrfi = 'SNR_combi_{}_{}_{}'.format(x1, color, z)

tab = load_multiple(dirFiles, snrfi, opts.sigmaC_min,
                    opts.sigmaC_max, opts.nproc)

"""
print('hello', tab.filter(regex='sigma').columns, len(tab))

save_file(tab)
"""

tab = tab.sort_values(by=['Nvisits'])
idx = tab['Nvisits'] < 100000000.
# idx &= tab['sigmaC'] >= 0.0390
snr = tab[idx]

snr = snr.rename(columns={'SNRcalc_tot': 'SNRcalc'})
print(snr.columns)
colors = dict(zip('grizy', 'bgrym'))
bands = 'rizy'
plot(snr, z, bands=bands, colors=colors)

# plot(snr, whata='Nvisits', whatb='SNRcalc', legy='$SNR_{band}$')

plot(snr, z, whata='sigmaC', whatb='SNRcalc',
     legx='sigmaC', legy='$SNR_{band}$', bands=bands, colors=colors)
plotb(snr, z, 'SNRcalc', 'Nvisits', bands=bands, colors=colors)
plotb(snr, z, 'SNRcalc', 'm5calc', bands=bands, colors=colors)
fig, ax = plt.subplots()
ax.plot(snr['sigmaC'], snr['Nvisits'], 'ko')


colout = ['sigmaC', 'SNRcalc_tot', 'Nvisits', 'Nvisits_g', 'SNRcalc_g',
          'Nvisits_r', 'SNRcalc_r', 'Nvisits_i', 'SNRcalc_i', 'Nvisits_z',
          'SNRcalc_z', 'Nvisits_y', 'SNRcalc_y', 'min_par', 'min_val']

colout = ['sigmaC', 'Nvisits', 'Nvisits_g', 'Nvisits_r', 'Nvisits_i', 'Nvisits_z',
          'Nvisits_y', 'min_par', 'min_val']

colout = ['sigmaC', 'Nvisits',
          'Nvisits_r', 'SNRcalc_r', 'Nvisits_i', 'SNRcalc_i', 'Nvisits_z',
                       'SNRcalc_z', 'Nvisits_y', 'SNRcalc_y']

sel = snr.copy()
"""
sel['Delta_iz'] = np.abs(sel['Nvisits_i']-sel['Nvisits_z'])
sel['Delta_SNR'] = sel['SNRcalc_z']-sel['SNRcalc_y']

seldict = {}
seldict['zmin'] = 0.6
seldict['cut1'] = {}
seldict['cut1']['var'] = 'Nvisits_r'
seldict['cut1']['value'] = 2
seldict['cut1']['op'] = operator.le
seldict['cut2'] = {}
seldict['cut2']['var'] = 'Nvisits_g'
seldict['cut2']['value'] = 2
seldict['cut2']['op'] = operator.le

seldictb = seldict.copy()
seldictb['cut3'] = {}
seldictb['cut3']['var'] = 'Delta_SNR'
seldictb['cut3']['value'] = 0.
seldictb['cut3']['op'] = operator.ge

selvar = ['Nvisits', 'Nvisits_y', 'Delta_iz']
minparname = ['nvisits', 'nvisits_y', 'deltav_iz']
"""
sel['Delta_SNR'] = sel['SNRcalc_z']-sel['SNRcalc_y']
sel['Delta_Nvisits'] = sel['Nvisits_z']-sel['Nvisits_y']
seldict = {}
seldict['zmin'] = 0.6
seldict['cut1'] = {}
seldict['cut1']['var'] = 'Nvisits_r'
seldict['cut1']['value'] = 2
seldict['cut1']['op'] = operator.le
seldict['cut2'] = {}
seldict['cut2']['var'] = 'Nvisits_g'
seldict['cut2']['value'] = 2
seldict['cut2']['op'] = operator.le

seldict['cut3'] = {}
seldict['cut3']['var'] = 'Delta_Nvisits'
seldict['cut3']['value'] = 3
seldict['cut3']['op'] = operator.ge

seldictb = seldict.copy()
seldictb['cut4'] = {}
seldictb['cut4']['var'] = 'Nvisits_y'
seldictb['cut4']['value'] = 20.
seldictb['cut4']['op'] = operator.le

seldictc = seldict.copy()
seldictc['cut4'] = {}
seldictc['cut4']['var'] = 'Nvisits_y'
seldictc['cut4']['value'] = 30.
seldictc['cut4']['op'] = operator.le

seldictd = seldict.copy()
seldictd['cut4'] = {}
seldictd['cut4']['var'] = 'Nvisits_y'
seldictd['cut4']['value'] = 40.
seldictd['cut4']['op'] = operator.le

selvar = ['Nvisits']
minparname = ['nvisits']


combi = dict(zip(selvar, minparname))
snr_visits = pd.DataFrame()

for key, val in combi.items():
    res = min_nvisits(z, sel, colout, key, val, seldict)

    print('parameter', key)
    # for key, val in combi.items():
    print('tttt', sel.columns)
    res = min_nvisits(z, sel, colout, key, '{}_sela'.format(val), seldictb)
    snr_visits = pd.concat((snr_visits, res))
    """
    # for key, val in combi.items():
    res = self.min_nvisits(sel, key, '{}_selb'.format(val), seldictc)
    snr_visits = pd.concat((snr_visits, res))

    # for key, val in combi.items():
    res = self.min_nvisits(sel, key, '{}_selc'.format(val), seldictd)
    snr_visits = pd.concat((snr_visits, res))
    """
print(snr_visits)
plt.show()
