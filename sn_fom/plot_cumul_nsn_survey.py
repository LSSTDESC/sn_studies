import pandas as pd
from __init__ import plt
import numpy as np
#from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d, make_interp_spline

lw = 3


def plot_cumulative(ax, dfa, dfb):

    dfa['nsn_cum'] = np.cumsum(resa['nsn'])/(np.cumsum(resa['nsn'])[-1])
    dfb['nsn_cum'] = np.cumsum(resb['nsn'])/(np.cumsum(resb['nsn'])[-1])
    dfa['nsn_cum'] = 1.-dfa['nsn_cum']
    dfb['nsn_cum'] = 1.-dfb['nsn_cum']
    dfa = dfa.to_records(index=False)
    dfb = dfb.to_records(index=False)

    zref = 0.8
    xa, ya, ia = smooth_it(dfa, 'z', 'nsn_cum', zval=zref)
    xb, yb, ib = smooth_it(dfb, 'z', 'nsn_cum', zval=zref)
    ax.plot(xa, ya, color='r', lw=lw, label='IDR$_{0.80}^{0.60}$')
    ax.plot([zref, zref], [0.0, ia], ls='dotted', color='k', lw=lw)
    ax.plot([0.0, zref], [ia, ia], ls='dotted', color='k', lw=lw)
    ax.text(0.2, ia+0.02, '{}'.format(np.round(ia, 2)), fontsize=20)
    ax.plot(xb, yb, color='b', lw=lw, label='DU$^{0.65}$', ls='dashed')
    ax.plot([zref, zref], [0.0, ib], ls='dotted', color='k', lw=lw)
    ax.plot([0.0, zref], [ib, ib], ls='dotted', color='k', lw=lw)
    ax.text(0.2, ib+0.02, '{}'.format(np.round(ib, 2)), fontsize=20)
    ax.grid()

    ax.set_ylabel('$N_{SN}^{frac} (\geq z)$')
    ax.set_xlabel('$z$')
    ax.set_xlim([0.1, 1.07])
    ax.set_ylim([0.0, 1.])
    # ax.legend(loc='upper left', bbox_to_anchor=(
    #    0., 1.15), ncol=2, frameon=False)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.15)


def plot_nsn(ax, dfa, dfb):

    dfa = dfa.to_records(index=False)
    dfb = dfb.to_records(index=False)

    zref = 0.8
    xa, ya, ia = smooth_it(dfa, 'z', 'nsn', zval=zref)
    xb, yb, ib = smooth_it(dfb, 'z', 'nsn', zval=zref)
    #ax.plot(dfa['z'], dfa['nsn'])
    #ax.plot(dfb['z'], dfb['nsn'])
    ax.plot(xa, ya, color='r', lw=lw, label='IDR$_{0.80}^{0.60}$')
    #ax.plot([zref, zref], [0.0, ia], ls='dotted', color='k', lw=lw)
    #ax.plot([0.0, zref], [ia, ia], ls='dotted', color='k', lw=lw)
    #ax.text(0.2, ia+0.02, '{}'.format(np.round(ia, 2)), fontsize=20)
    ax.plot(xb, yb, color='b', lw=lw, label='DU$^{0.65}$', ls='dashed')
    #ax.plot([zref, zref], [0.0, ib], ls='dotted', color='k', lw=lw)
    #ax.plot([0.0, zref], [ib, ib], ls='dotted', color='k', lw=lw)
    #ax.text(0.2, ib+0.02, '{}'.format(np.round(ib, 2)), fontsize=20)
    ax.grid()

    ax.set_ylabel('$N_{SN}$')
    ax.set_xlabel('$z$')
    ax.set_xlim([0.1, 1.07])
    ax.set_ylim([0.0, None])
    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.35), ncol=2, frameon=False)
    # fig.tight_layout()
    plt.subplots_adjust(left=0.15)
    ax.set_xticklabels([])


def smooth_it(vals, xvar, yvar, zval=0.80):
    xmin, xmax = np.min(vals[xvar]), np.max(vals[xvar])
    xnew = np.linspace(xmin, xmax, 100)
    spl = make_interp_spline(
        vals[xvar], vals[yvar], k=3)  # type: BSpline
    """
    spl = UnivariateSpline(vals[xvar], vals[yvar], k=5)
    spl.set_smoothing_factor(0.5)
    """
    spl_smooth = spl(xnew)

    ii = interp1d(xnew, spl_smooth, bounds_error=False, fill_value=0.)
    return xnew, spl_smooth, ii(zval)


resa = pd.read_hdf('edr_0.80_4.hdf5').to_records(index=False)
resb = pd.read_hdf('universal_3.hdf5').to_records(index=False)

dfa = pd.DataFrame(resa)
dfb = pd.DataFrame(resb)

fig, ax = plt.subplots(figsize=(9, 16), nrows=2)
print(np.cumsum(resa['nsn']))
plot_nsn(ax[0], dfa, dfb)
plot_cumulative(ax[1], dfa, dfb)

plt.subplots_adjust(hspace=0.04)
plt.show()
