import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import glob
from optparse import OptionParser


def load(thedir, snrfi):
    """
    Method to load and concatenate a set of npy files

    Parameters
    ---------------
    thedir: str
       location directory of the file
    snrfi: str
       name of the files (without npy extension)

    Returns
    -----------
    tab: pandas df
      data

    """
    fi = glob.glob('{}/{}_*.npy'.format(thedir, snrfi))

    snrtot = None
    for ff in fi:
        rr = np.load(ff, allow_pickle=True)
        if snrtot is None:
            snrtot = rr
        else:
            snrtot = np.concatenate((snrtot, rr))

    tab = pd.DataFrame(snrtot)

    return tab


def plotb(tab, whata='Nvisits', whatb='Nvisits', leg='$N_{visits}$'):
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

    """

    fontsize = 15
    fig, ax = plt.subplots()
    cadence = '1 day$^{-1}$'
    sigmaC = '$\sigma_{color} \sim 0.04\pm1\%$'
    fig.suptitle('($x_1,color$)=({},{}) - z={} \n cadence={} - {}'.format(-2.0,
                                                                          0.2, 0.7, cadence, sigmaC), fontsize=fontsize)
    ax.plot(tab['{}_i'.format(whata)],
            tab['{}_i'.format(whatb)], 'y.', label='$i$-band')
    ax.plot(tab['{}_z'.format(whata)],
            tab['{}_z'.format(whatb)], 'r.', label='$z$-band')
    ax.plot(tab['{}_y'.format(whata)],
            tab['{}_y'.format(whatb)], 'm.', label='$y$-band')

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


def plot(tab, whata='Nvisits', whatb='Nvisits', legx='$N_{visits}$', legy='$N_{visits}^{band}$'):
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

    """

    fontsize = 15
    fig, ax = plt.subplots()
    cadence = '1 day$^{-1}$'
    sigmaC = '$\sigma_{color} \sim 0.04\pm1\%$'
    fig.suptitle('($x_1,color$)=({},{}) - z={} \n cadence={} - {}'.format(-2.0,
                                                                          0.2, 0.7, cadence, sigmaC), fontsize=fontsize)
    ax.plot(tab[whata], tab['{}_i'.format(whatb)], 'y.', label='$i$-band')
    ax.plot(tab[whata], tab['{}_z'.format(whatb)], 'r.', label='$z$-band')
    ax.plot(tab[whata], tab['{}_y'.format(whatb)], 'm.', label='$y$-band')

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


opts, args = parser.parse_args()

dirFiles = opts.dirFiles
z = np.round(opts.z, 2)
x1 = np.round(opts.x1, 1)
color = np.round(opts.color, 1)
snrfi = 'SNR_combi_{}_{}_{}'.format(x1, color, z)

tab = load(dirFiles, snrfi)


print('hello', tab.filter(regex='sigma').columns)

tab = tab.rename(columns={'SNRcalc_tot': 'SNRcalc'})
print(tab.columns)


plot(tab)

# plot(tab, whata='Nvisits', whatb='SNRcalc', legy='$SNR_{band}$')

plot(tab, whata='sigmaC', whatb='SNRcalc', legx='sigmaC', legy='$SNR_{band}$')
plotb(tab, 'SNRcalc', 'Nvisits')

plt.show()
