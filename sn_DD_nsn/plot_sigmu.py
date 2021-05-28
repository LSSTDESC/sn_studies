from sn_DD_nsn import plt, filtercolors
from optparse import OptionParser
import h5py
from sn_tools.sn_io import loopStack
#import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def loadFile(fileDir, fileName):
    fullName = '{}/{}'.format(fileDir, fileName)

    sn = loopStack([fullName], 'astropyTable')

    return sn


def plot_errorbud_z(sna, snb):

    fig, ax = plt.subplots(figsize=(10, 7))

    plot(ax, sna)
    if snb is not None:
        plot(ax, snb)


def plot(ax, sna):

    idx = sna['z'] <= 0.8
    sn = sna[idx]
    sn.sort('z')
    #ax.plot(sn['z'], sn['sigma_mu'], color='k',label='$\sigma_{\mu}$')
    alph = '$\\alpha$'
    bet = '$\\beta$'
    ax.plot(sn['z'], sn['beta']*np.sqrt(sn['Cov_colorcolor']),
            color='r', label=bet+'$\sigma_C$')
    ax.plot(sn['z'], sn['alpha']*np.sqrt(sn['Cov_x1x1']),
            color='b', label=alph+'$\sigma_{x1}$')
    ax.plot(sn['z'], np.sqrt(sn['Cov_mbmb']),
            color='g', label='$\sigma_{m_b}$')

    miny = np.min(sn['alpha']*np.sqrt(sn['Cov_x1x1']))
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xlim([0.3, 0.8])
    ax.set_ylim([0., None])
    ax.set_xlabel(r'z')
    ax.set_ylabel(r'Error budget [mag]')

    interp = interp1d(sn['beta']*np.sqrt(sn['Cov_colorcolor']), sn['z'])
    zlim = interp(0.12)
    ax.plot([0.3, zlim], [0.12]*2, color='k', lw=1, ls='dashed')
    ax.plot([zlim]*2, [0., 0.12], color='k', lw=1, ls='dashed')
    zlimstr = '$z_{limit}$'
    ax.text(0.63, 0.125, zlimstr+' = {}'.format(np.round(zlim, 2)), fontsize=15)


def plot_snr_sigmaC(sna):

    print(sna.columns)
    fig, ax = plt.subplots(figsize=(10, 7))

    sna.sort('z')
    bands = 'izy'
    snrmin = {}
    for b in bands:
        ax.plot(sna['SNR_{}'.format(b)], np.sqrt(sna['Cov_colorcolor']),
                color=filtercolors[b], label='{} band'.format(b))
        ii = interp1d(np.sqrt(sna['Cov_colorcolor']), sna['SNR_{}'.format(b)])
        snrmin[b] = ii(0.04)

    ax.set_xlim([0., 80.])
    ax.set_ylim([0., 0.06])
    ax.set_xlabel(r'Signal-to-Noise Ratio')
    ax.set_ylabel(r'$\sigma_C$')
    ax.grid()
    ax.legend()

    for b in bands:
        ax.plot([0., snrmin[b]], [0.04]*2, color='k', lw=1, ls='dashed')
        ax.plot([snrmin[b]]*2, [0., 0.04], color='k', lw=1, ls='dashed')
        ffi = 'SNR$_{'+b+'}$'
        ax.text(snrmin[b]-10, 0.041, '{} = {}'.format(ffi,
                                                      np.round(snrmin[b])), color=filtercolors[b], fontsize=15)


parser = OptionParser()

parser.add_option(
    '--fileDir', help='file directory [%default]', default='zlim_test/Output_Fit_error_model_ebvofMW_0.0_snrmin_1_errmodrel_0.1', type=str)
parser.add_option(
    '--fileDirb', help='file directory [%default]', default='zlim_test/Output_Fit_error_model_ebvofMW_0.0_snrmin_1_errmodrel_0.1', type=str)
parser.add_option(
    '--fileName', help='file to process [%default]', default='Fit_sn_cosmo_Fake_-2.0_0.2_error_model_ebvofMW_0.0_1_sn_cosmo.hdf5', type=str)
parser.add_option(
    '--fileNameb', help='2nd file to process [%default]', default='Fit_sn_fast_Fake_-2.0_0.2_error_model_ebvofMW_0.0_0_sn_fast.hdf5', type=str)


opts, args = parser.parse_args()

fileDir = opts.fileDir
fileName = opts.fileName
fileNameb = opts.fileNameb

sn = loadFile(fileDir, fileName)
sna = loadFile(opts.fileDirb, fileName.replace('error_model', '380.0_800.0'))
snb = loadFile(fileDir, fileNameb)

plot_errorbud_z(sn, None)
plot_snr_sigmaC(snb)


plt.show()
