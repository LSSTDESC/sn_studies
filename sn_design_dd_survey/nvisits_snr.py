import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from scipy.interpolate import griddata, interp2d
from scipy.ndimage.filters import gaussian_filter

colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))


def plot_contour_Nvisits(data):
    """
    Method to make a contour plot of Nvisits in the plane(SNR,z)

    Parameters
    ---------------
    data_orig: pandas df
    data to process

    """

    fig, ax = plt.subplots()
    zmin, zmax = 0.5, 1.0
    SNRmin, SNRmax = 0., 100.

    zvals = np.linspace(zmin, zmax, 1000)
    SNRvals = np.linspace(SNRmin, SNRmax, 100)

    print('meshgrid')
    Z, SNR = np.meshgrid(zvals, SNRvals)

    color = 'k'
    ls = 'dashed'
    for b in 'izy':
        idx = data['band'] == b
        sel = data[idx]
        print('interpol instance')
        x = sel['z'].to_list()
        y = sel['SNR'].to_list()
        #v = np.tile(sel['nVisits'], (len(y), 1))
        v = sel['nVisits'].to_list()
        #v = fillZ(sel.to_records(index=False), 'z', 'SNR', 'nVisits')
        """
        print(x)
        print(y)
        print(v)
        print(sel['SNR'])
        print(len(x), len(y), v.shape, Z.shape, SNR.shape)
        interpol = interp2d(
            x, y, v, bounds_error=False, fill_value=0.)
        print('interpol values')
        NV = interpol(zvals, SNRvals)
        print(NV)
        print(Z)
        print(SNR)
        """
        NV = griddata((x, y), v, (Z, SNR), method='linear')
        # ax.imshow(NV, extent=(
        #    zmin, zmax, SNRmin, SNRmax), aspect='auto', alpha=0.25, cmap='hsv')
        print('plotting contour')
        nvv = range(10, 100, 20)
        CS = plt.contour(Z, SNR, gaussian_filter(
            NV, sigma=3.), nvv, colors=colors[b])
        fmt = {}
        strs = ['%i' % nn for nn in nvv]
        print(strs)
        #strs = ['{}%'.format(np.int(zz)) for zz in zzvc]
        for l, s in zip(CS.levels, strs):
            fmt[l] = s
        ax.clabel(CS, inline=True, fontsize=10,
                  colors=colors[b], fmt=fmt)

        # plt.show()


def plot_SNR_Nvisits(sel_snr, bands='izy', z=np.arange(0.8, 1.1, 0.1)):
    """
    function to plot SNR vs nvisits vs SNR for bands 

    Parameters
    --------------
    sel_snr: pandas df
      data to plot
    bands: str, opt
      bands to plot (default: iry)
    z: array, opt
      redshift range to plot (default: np.arange(0.8,1.1,0.1)

    """
    ls = dict(zip(z, ['solid', 'dotted', 'dashed', 'solid']))
    print(ls, z)
    fig, ax = plt.subplots()
    for b in bands:
        ida = sel_snr['band'] == b
        sela = sel_snr[ida]
        for zv in z:
            if zv <= 1:
                idb = np.abs(sela['z']-zv) < 1.e-5
                selb = sela[idb].to_records(index=False)
                ax.plot(selb['nVisits'], selb['SNR_photo_bd'], color=colors[b],
                        label='{} - z={}'.format(b, zv), ls=ls[zv])

    ax.grid()
    ax.set_xlabel('Nvisits')
    ax.set_ylabel('SNR')
    ax.legend()
    ax.set_xlim([0., None])
    ax.set_ylim([0., None])


def plot_z(zref, sel_snr):
    """
    function to plot nvisits vs SNR and SNR vs m5 for a defined redshift

    Parameters
    --------------
    zref: float
      the redshift of reference
    sel_snr: pandas df
      data to plot

    """

    fig, ax = plt.subplots()
    figb, axb = plt.subplots()
    colors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
    for band in np.unique(sel_snr['band']):
        idx = sel_snr['band'] == band
        idx &= np.abs(sel_snr['z']-zref) < 1.e-5
        sel_snrb = sel_snr[idx].to_records(index=False)

        ax.plot(sel_snrb['nVisits'], sel_snrb['SNR'],
                color=colors[band], label=band)
        ax.plot(sel_snrb['nVisits'], sel_snrb['SNR_photo_bd'],
                ls='--', color=colors[band])
        axb.plot(sel_snrb['SNR'], sel_snrb['m5'],
                 color=colors[band], label=band)

    ax.legend()
    ax.set_xlabel('nVisits')
    ax.set_ylabel('SNR')
    ax.grid()
    axb.legend()
    axb.set_xlabel('SNR')
    axb.set_ylabel('m5')
    axb.grid()


def plot_SNRmax(sel_snr, what='SNR'):
    """
    function to plot snrmax vs z

    Parameters
    --------------
    sel_snr: pandas df
      data to plot
    what: str, opt
      what to plot (default: SNR)

    """

    selb = sel_snr.groupby(['band', 'z'])[what].apply(
        lambda x: x.max()).reset_index()

    fig, ax = plt.subplots()

    for b in np.unique(selb['band']):
        iu = selb['band'] == b
        al = selb[iu].to_records(index=False)
        ax.plot(al['z'], al[what], label=b)

    ax.legend()
    ax.grid()


parser = OptionParser()

parser.add_option("--dirFiles", type="str", default='dd_design_faint_errorcut/SNR_m5',
                  help="location dir of (m5,SNR) files [%default]")
parser.add_option("--snrFile", type="str", default='SNR_m5_sn_fast_-2.0_0.2_ebv_0.0_error_model_cad_1_0_0.1.npy',
                  help="(SNR,m5) file name [%default]")
parser.add_option("--m5File", type="str", default='input/sn_studies/medValues_flexddf_v1.4_10yrs_DD.npy',
                  help="file to get m5 reference single band values[%default]")
parser.add_option("--z", type=str, default=0.8,
                  help="list of redshift value for display[%default]")

opts, args = parser.parse_args()

snr_file = '{}/{}'.format(opts.dirFiles, opts.snrFile)

snr = pd.DataFrame(np.load(snr_file, allow_pickle=True))

m5_file = opts.m5File

m5 = pd.DataFrame(np.load(m5_file, allow_pickle=True))

med_m5 = m5.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()

print(med_m5)

snr = snr.merge(med_m5, left_on=['band'], right_on=['filter'])

snr = snr.rename(columns={"fiveSigmaDepth": "m5_single"})

snr['nVisits'] = 10**((snr['m5']-snr['m5_single'])/1.25)
print(snr)

io = snr['nVisits'] <= 250.  # max number of visits per band

sel_snr = snr[io]

# plot_SNRmax(sel_snr,)

zref = list(map(float, opts.z.split(',')))
# plot_z(zref,sel_snr)

plot_SNR_Nvisits(sel_snr, z=zref)
plot_contour_Nvisits(sel_snr)

plt.show()
