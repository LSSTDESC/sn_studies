import glob
import pandas as pd
import numpy as np
from sn_fom import plt
from optparse import OptionParser
from scipy.interpolate import make_interp_spline, BSpline, griddata
import os
from scipy.ndimage.filters import gaussian_filter


def getVals(res, varx='zcomp', vary='sigma_w', varz='nddf', nbins=800, method='linear'):

    xmin, xmax = res[varx].min(), res[varx].max()
    xlim = np.linspace(xmin, xmax, nbins)
    ymin, ymax = res[vary].min(), res[vary].max()
    ylim = np.linspace(ymin, ymax, nbins)

    X, Y = np.meshgrid(xlim, ylim)
    # X, Y = xlim,ylim
    X_grid = np.c_[np.ravel(X), np.ravel(Y)]
    # Z = griddata((res[varx],res[vary]),res[varz],(X,Y),method=method)
    Z = griddata((res[varx], res[vary]), res[varz], X_grid, method=method)
    Z = Z.reshape(X.shape)
    return X, Y, Z


def make_summary(fis, cosmo_scen, runtype='deep_rolling'):

    r = []
    for fi in fis:
        # get config name
        conf = fi.split('/')[-1].split('.hdf5')[0]
        conf = '_'.join(conf.split('_')[1:])
        params_fit = pd.read_hdf(fi)
        idx = params_fit['accuracy'] == 1
        params_fit = params_fit[idx]
        # print(params_fit)
        mean = np.median(params_fit['sigma_w0'])
        std = np.std(params_fit['sigma_w0'])
        nsn_DD = np.median(params_fit['nsn_DD'])
        idxb = cosmo_scen['configName'] == conf
        scen = cosmo_scen[idxb]
        if len(scen) > 0:
            zcomp, nddf = decode_scen(scen, runtype=runtype)
            if nddf > 0:
                r.append((conf, mean, std, zcomp, nddf, nsn_DD))

    res = pd.DataFrame(
        r, columns=['conf', 'sigma_w', 'sigma_w_std', 'zcomp', 'nddf', 'nsn_DD'])

    print(res[['conf', 'sigma_w', 'zcomp', 'nddf', 'nsn_DD']])

    return res


def decode_scen(scen, runtype='deep_rolling'):

    dbName = scen['dbName'].to_list()[0]
    fields = scen['fields'].to_list()[0]
    pointings = scen['npointings'].to_list()[0]

    zcomp = -1
    nddf = []

    if '/' in dbName:

        dbName = dbName.split('/')[-1]
        zcomp = float(dbName.split('_')[-1])

        nddf = pointings.split('/')[-1].split(',')
        nddf = list(map(int, nddf))
    else:
        if runtype != 'deep_rolling':
            zcomp = float(dbName.split('_')[-1])

            nddf = pointings.split(',')
            nddf = list(map(int, nddf))
        else:
            zcomp = 0.
            nddf = 0

    return zcomp, int(np.sum(nddf))


def smooth(res, plot=False):

    df_smooth = pd.DataFrame()
    fig = None
    if plot:
        fig, ax = plt.subplots()
    for nddf in res['nddf'].unique():
        print('processing', nddf)
        idx = res['nddf'] == nddf
        sel = res[idx]
        sel = sel.sort_values(by=['zcomp'])
        if fig:
            print(sel)
            ax.plot(sel['zcomp'], sel['sigma_w'])
            plt.show()

        xnew = np.linspace(np.min(sel['zcomp']), np.max(sel['zcomp']), 100)
        spl = make_interp_spline(
            sel['zcomp'], sel['sigma_w'], k=3)  # type: BSpline
        spl_smooth = spl(xnew)
        splb = make_interp_spline(sel['zcomp'], sel['nsn_DD'], k=3)
        splb_smooth = splb(xnew)
        # power_smooth = spline(sel['zcomp'], sel['sigma_w'], xnew)
        if fig:
            ax.plot(xnew, splb_smooth)
        # print(xnew,power_smooth)
        dda = pd.DataFrame(xnew, columns=['zcomp'])
        dda['sigma_w'] = spl_smooth
        dda['nsn_DD'] = splb_smooth
        dda['nddf'] = nddf
        dda = dda.round({'nsn_DD': 1})
        df_smooth = pd.concat((df_smooth, dda))

    return df_smooth


parser = OptionParser()

parser.add_option('--fileDir', type=str, default='/sps/lsst/users/gris/fake/Fit_new/',
                  help='file directory [%default]')
parser.add_option('--config', type=str, default='config_cosmoSN_dr_0.9.csv',
                  help='config file  [%default]')
parser.add_option('--runtype', type=str, default='deep_rolling',
                  help='runtype (deep_rolling/universal) [%default]')

opts, args = parser.parse_args()

confName = opts.config.split('.')[0].split('FitParams_conf_')[-1]
confName = '_'.join(vv for vv in confName.split('_')[2:])

fis = glob.glob('{}/*{}*.hdf5'.format(opts.fileDir, confName))
cosmo_scen = pd.read_csv(opts.config, delimiter=';', comment='#')

outName = opts.config.replace('config_', '').replace('csv', 'hdf5')

if not os.path.isfile(outName):
    print('moving to summ')
    res = make_summary(fis, cosmo_scen, runtype=opts.runtype)
    res.to_hdf(outName, key='cosmo')

res = pd.read_hdf(outName)

res = smooth(res, plot=True)


fig, ax = plt.subplots()
axb = ax.twinx()

for nddf in res['nddf'].unique():
    idx = res['nddf'] == nddf
    sel = res[idx]
    ax.plot(sel['zcomp'], sel['sigma_w'], color='k')
    axb.plot(sel['zcomp'], sel['nsn_DD'], ls='dashed', color='r')

fig, ax = plt.subplots(figsize=(12, 8))
ZLIMIT, NDDF, SIGMAS = getVals(res, 'zcomp', 'nddf', 'sigma_w', nbins=500)
ax.imshow(SIGMAS, origin='lower', extent=(1., 6., 0.65, 0.90),
          aspect='auto', alpha=0.25, cmap='hsv')
zzv = [0.010, 0.012, 0.013, 0.014, 0.015, 0.016]
CS = ax.contour(NDDF, ZLIMIT, SIGMAS, zzv, colors='k')
fmt = {}
strs = ['$%3.3f$' % zz for zz in zzv]
# strs = ['{}%'.format(np.int(zz)) for zz in zzvc]
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax.clabel(CS, inline=True, fontsize=15,
          colors='k', fmt=fmt)

ZLIMITB, NDDF_NSN, NSN_DD = getVals(
    res, 'zcomp', 'nddf',  'nsn_DD', nbins=800, method='linear')
# zzv = [1000., 1500., 2000., 2500., 3000.]
zzv = [3000., 4000., 6000., 8000., 10000., 12000.]
CSb = ax.contour(NDDF_NSN, ZLIMITB, NSN_DD, zzv,
                 colors='r', linestyles='dashed')
fmt = {}
# strs = ['$%3.3f$' % zz for zz in zzv]
strs = ['{}'.format(np.int(zz)) for zz in zzv]
for l, s in zip(CSb.levels, strs):
    fmt[l] = s
axb.clabel(CSb, inline=True, fontsize=15,
           colors='r', fmt=fmt)

ax.grid()
"""

ZLIMIT, SIGMAS, NDDF = getVals(res, 'zcomp', 'sigma_w', 'nddf', nbins=500)
ZLIMITB, NSN_DD, NDDF_NSN = getVals(
    res, 'zcomp', 'nsn_DD', 'nddf', nbins=500, method='linear')

fig, ax = plt.subplots(figsize=(12, 8))
# ax.imshow(NDDF, origin='lower',extent=(0.65,0.9,0.001, 0.05),
#          aspect='auto', alpha=0.25, cmap='hsv')

zzv = [1., 2., 3., 3.98]
CS = ax.contour(ZLIMIT, SIGMAS, NDDF, zzv, colors='k')
# CS = ax.contour(ZLIMIT, NSN_DD, NDDF_NSN, zzv, colors='k')

fmt = {}
strs = ['N$_{DD}$=$%3.0f$' % zz for zz in zzv]
# strs = ['{}%'.format(np.int(zz)) for zz in zzvc]
for l, s in zip(CS.levels, strs):
    fmt[l] = s
manual_locations = [(0.72, 0.01752), (0.72, 0.0156),
                    (0.72, 0.01425), (0.72, 0.013380)]
ax.clabel(CS, inline=True, fontsize=15,
          colors='r', fmt=fmt, manual=manual_locations, inline_spacing=1, use_clabeltext=True)

axb = ax.twinx()

strs = ['N$_{DD}$=$%3.0f$' % zz for zz in zzv]
CSb = axb.contour(ZLIMITB, NSN_DD, NDDF_NSN, zzv,
                  colors='k', linestyles='dashed')
for l, s in zip(CS.levels, strs):
    fmt[l] = s
manual_locations = [(0.68, 2786), (0.68, 2320),
                    (0.68, 1826), (0.68, 1334.)]
axb.clabel(CSb, inline=True, fontsize=15,
           colors='b', fmt=fmt, manual=manual_locations)
ax.grid()
ax.set_xlabel('$z_{complete}$')
ax.set_ylabel('$\sigma_w$')
axb.set_ylabel('N$_{SN}$')
"""
"""
# Fluxes and errors
zmin, zmax, zstep, nz = limVals(res, 'zcomp')
phamin, phamax, phastep, npha = limVals(res, 'sigma_w')
print('alors',phamin, phamax, phastep, npha)
zstep = np.round(zstep, 2)
phastep = np.round(phastep, 3)

zv = np.linspace(zmin, zmax, nz)
phav = np.linspace(phamin, phamax, npha)

index = np.lexsort((res['zcomp'], res['sigma_w']))
nddf = np.reshape(res[index]['nddf'], (npha, nz))

ddf_resu = RegularGridInterpolator(
    (phav, zv), nddf, method='linear', bounds_error=False, fill_value=-1.0)

# ready to make the plot

zlim = np.linspace(zmin, zmax, 100)
sigma_w = np.linspace(phamin, phamax, 100)

ZLIMIT, SIGMAS = np.meshgrid(zlim,sigma_w)

NDDF = ddf_resu((SIGMAS,ZLIMIT))

fig, ax = plt.subplots()
ax.imshow(NDDF, extent=(phamin, phamax, zmin, zmax),
          aspect='auto', alpha=0.25, cmap='hsv')

zzv = [1., 2.,3., 4.]
CS = ax.contour(SIGMAS, ZLIMIT, NDDF, zzv, colors='k')

fmt = {}
strs = ['$%3.2f$' % zz for zz in zzv]
# strs = ['{}%'.format(np.int(zz)) for zz in zzvc]
for l, s in zip(CS.levels, strs):
    fmt[l] = s
ax.clabel(CS, inline=True, fontsize=10,
          colors='k', fmt=fmt)
"""
plt.show()
