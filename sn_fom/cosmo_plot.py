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


def to_string(ll):

    return ll
    print('allo', ll)
    return '_'.join(['{}'.format(l) for l in ll])


def make_summary(fis, cosmo_scen, runtype='deep_rolling'):

    r = []
    fields = ['COSMOS', 'XMM-LSS', 'ADFS', 'ELAIS', 'CDFS']
    print('hello man', fis)
    for fi in fis:
        nFields = {}
        # get config name
        conf = fi.split('/')[-1].split('.hdf5')[0]
        conf = '_'.join(conf.split('_')[1:])
        params_fit = pd.read_hdf(fi)
        idx = params_fit['accuracy'] == 1
        params_fit = params_fit[idx]
        # print(params_fit.columns)
        sigma_w = np.median(params_fit['sigma_w0'])
        sigma_Om = np.median(params_fit['sigma_Om'])
        mean_w = np.median(params_fit['w0'])
        mean_Om = np.median(params_fit['Om'])
        std = np.std(params_fit['sigma_w0'])
        nsn_DD = np.median(params_fit['nsn_DD'])
        for field in fields:
            bb = 'nsn_DD_{}'.format(field)
            if bb in params_fit.columns:
                nFields[bb] = np.median(params_fit[bb])
            else:
                nFields[bb] = 0.
        nsn_z_09 = check_get(params_fit, 'nsn_z_09')
        nsn_ultra = check_get(params_fit, 'nsn_ultra')
        nsn_ultra_z_08 = check_get(params_fit, 'nsn_ultra_z_08')
        nsn_dd = check_get(params_fit, 'nsn_dd')
        nsn_dd_z_05 = check_get(params_fit, 'nsn_dd_z_05')
        nsn_spectro_ultra_yearly = check_get(
            params_fit, 'nsn_spectro_ultra_yearly')
        nsn_spectro_ultra_tot = check_get(params_fit, 'nsn_spectro_ultra_tot')
        nsn_spectro_deep_yearly = check_get(
            params_fit, 'nsn_spectro_deep_yearly')
        nsn_spectro_deep_tot = check_get(params_fit, 'nsn_spectro_deep_tot')

        idxb = cosmo_scen['configName'] == conf
        scen = cosmo_scen[idxb]
        print('hhh', scen)
        if len(scen) > 0:
            """
            zcomp, zcomp_ultra, nddf, nddf_ultra, nseasons_ultra = decode_scen(
                scen, runtype=runtype)
             if nddf > 0:
                r.append((conf, mean, std, zcomp, zcomp_ultra,
                          nddf, nddf_ultra, nsn_DD, nseasons_ultra))
            """
            ddf_dd, zcomp_dd, nseasons_dd, ddf_ultra, zcomp_ultra, nseasons_ultra, year = decode_scen(
                scen, runtype=runtype)
            bn = [conf, mean_Om, sigma_Om, mean_w, sigma_w, std, ddf_dd, zcomp_dd,
                  nseasons_dd, ddf_ultra, zcomp_ultra, nseasons_ultra, nsn_DD, nsn_z_09, nsn_ultra, nsn_ultra_z_08, nsn_dd, nsn_dd_z_05, year]
            bn += [nsn_spectro_ultra_yearly, nsn_spectro_ultra_tot,
                   nsn_spectro_deep_yearly, nsn_spectro_deep_tot]
            for field in fields:
                bn += [nFields['nsn_DD_{}'.format(field)]]
            r.append(tuple(bn))
    """
    res = pd.DataFrame(
        r, columns=['conf', 'sigma_w', 'sigma_w_std', 'zcomp', 'zcomp_ultra', 'nddf', 'nddf_ultra', 'nsn_DD', 'nseasons_ultra'])

    print(res[['conf', 'sigma_w', 'zcomp', 'nddf',
               'nsn_DD', 'zcomp_ultra', 'nddf', 'nddf_ultra', 'nseasons_ultra','year']])
    """
    colfields = []
    for field in fields:
        colfields += ['nsn_DD_{}'.format(field)]

    ccols = ['conf', 'Om', 'sigma_Om', 'w', 'sigma_w', 'sigma_w_std', 'ddf_dd', 'zcomp_dd',
             'nseasons_dd', 'ddf_ultra', 'zcomp_ultra', 'nseasons_ultra', 'nsn_DD', 'nsn_z_09', 'nsn_ultra', 'nsn_ultra_z_08', 'nsn_dd', 'nsn_dd_z_05', 'year']

    ccols += ['nsn_spectro_ultra_yearly', 'nsn_spectro_ultra_tot',
              'nsn_spectro_deep_yearly', 'nsn_spectro_deep_tot']
    ccols += colfields

    res = pd.DataFrame(
        r, columns=ccols)
    return res


def check_get(params_fit, varname):

    resu = 0.
    if varname in params_fit.columns:
        resu = np.median(params_fit[varname])

    return resu


def decode_scen_deprecated(scen, runtype='deep_rolling'):

    dbName = scen['dbName'].to_list()[0]
    fields = scen['fields'].to_list()[0]
    pointings = scen['npointings'].to_list()[0]
    seasons = scen['nseasons'].to_list()[0]
    print('here is the scene', scen)
    zcomp = -1
    nddf = []

    nddf_ultra = 0
    zcomp_ultra = 0
    zcomp = 0.
    nddf = 0
    nddf_ultra = 0
    zcomp_ultra = 0
    nseasons_ultra = 0

    if '/' in dbName:

        dbNamespl = dbName.split('/')
        zcomp = float(dbNamespl[-1].split('_')[-1])
        zcomp_ultra = float(dbNamespl[0].split('_')[-1])

        nddf = pointings.split('/')[-1].split(',')
        nddf_ultra = pointings.split('/')[0].split(',')
        nseasons_ultra = seasons.split('/')[0].split(',')
        fields_ultra = fields.split('/')[0].split(',')
        fields_dd = fields.split('/')[1].split(',')

        nddf = list(map(int, nddf))
        nddf_ultra = list(map(int, nddf_ultra))
        nseasons_ultra = list(map(int, nseasons_ultra))
        print('allo', nddf, nddf_ultra, nseasons_ultra,
              [zcomp_ultra]*np.sum(nddf_ultra), fields_ultra, fields_dd)
    else:
        if runtype != 'deep_rolling':
            zcomp = float(dbName.split('_')[-1])

            nddf = pointings.split(',')
            nddf = list(map(int, nddf))
        else:
            zcomp_ultra = float(dbName.split('_')[-1])
            nddf_ultra = pointings.split(',')
            nseasons_ultra = seasons.split(',')
            nddf_ultra = list(map(int, nddf_ultra))
            nseasons_ultra = list(map(int, nseasons_ultra))

    nddf = int(np.sum(nddf))
    nddf_ultra = int(np.sum(nddf_ultra))
    nseasons_ultra = int(np.median(nseasons_ultra))
    return zcomp, zcomp_ultra, nddf, nddf_ultra, nseasons_ultra


def decode_scen(scen, runtype='deep_rolling'):

    dbName = scen['dbName'].to_list()[0]
    dbNamespl = dbName.split('/')
    fields = scen['fields'].to_list()[0]
    pointings = scen['npointings'].to_list()[0]
    seasons = scen['nseasons'].to_list()[0]
    year = scen['year'].to_list()[0]
    print('here is the scene', scen)
    zcomp = -1
    nddf = []

    nddf_ultra = 0
    zcomp_ultra = 0
    zcomp = 0.
    nddf = 0
    nddf_ultra = 0
    zcomp_ultra = 0
    nseasons_ultra = 0

    ddf_ultra = []
    zcomp_ultra = []
    nseasons_ultra = []

    ddf_dd = []
    zcomp_dd = []
    nseasons_dd = []
    if '/' in dbName:

        # ultra deep fields
        zcomp_ultra = float(dbNamespl[0].split('_')[-1])
        nddf_ultra = pointings.split('/')[0].split(',')
        nseasons_ultra = seasons.split('/')[0].split(',')
        ddf_ultra = fields.split('/')[0].split(',')
        nddf_ultra = list(map(int, nddf_ultra))
        zcomp_ultra = [zcomp_ultra]*np.sum(nddf_ultra)
        nseasons_ultra = list(map(int, nseasons_ultra))

        # deep fields
        zcomp = float(dbNamespl[-1].split('_')[-1])
        nddf = list(map(int, pointings.split('/')[-1].split(',')))
        nseasons = list(map(int, seasons.split('/')[1].split(',')))
        fields_dd = fields.split('/')[1].split(',')

        for i in range(len(nddf)):
            ddf_dd += [fields_dd[i]]*nddf[i]
            zcomp_dd += [zcomp]*nddf[i]
            nseasons_dd += [nseasons[i]]*nddf[i]

        print('allo', ddf_dd, zcomp_dd, nseasons_dd,
              ddf_ultra, zcomp_ultra, nseasons_ultra)
    else:
        if runtype != 'deep_rolling':
            zcomp = float(dbName.split('_')[-1])

            nddf = pointings.split(',')
            nddf = list(map(int, nddf))
            # deep fields
            zcomp = float(dbNamespl[-1].split('_')[-1])
            nddf = list(map(int, pointings.split('/')[-1].split(',')))
            nseasons = list(map(int, seasons.split('/')[0].split(',')))
            fields_dd = fields.split('/')[0].split(',')

            for i in range(len(nddf)):
                ddf_dd += [fields_dd[i]]*nddf[i]
                zcomp_dd += [zcomp]*nddf[i]
                nseasons_dd += [nseasons[i]]*nddf[i]

        else:
            zcomp_ultra = float(dbName.split('_')[-1])
            nddf_ultra = pointings.split(',')
            nseasons_ultra = seasons.split(',')
            nddf_ultra = list(map(int, nddf_ultra))
            nseasons_ultra = list(map(int, nseasons_ultra))

            # ultra deep fields
            zcomp_ultra = float(dbNamespl[0].split('_')[-1])
            nddf_ultra = pointings.split('/')[0].split(',')
            nseasons_ultra = seasons.split('/')[0].split(',')
            ddf_ultra = fields.split('/')[0].split(',')
            nddf_ultra = list(map(int, nddf_ultra))
            zcomp_ultra = [zcomp_ultra]*np.sum(nddf_ultra)
            nseasons_ultra = list(map(int, nseasons_ultra))
            """
            nddf = int(np.sum(nddf))
            nddf_ultra = int(np.sum(nddf_ultra))
            nseasons_ultra = int(np.median(nseasons_ultra))
            """
    # return zcomp, zcomp_ultra, nddf, nddf_ultra, nseasons_ultra
    return to_string(ddf_dd), to_string(zcomp_dd), to_string(nseasons_dd), to_string(ddf_ultra), to_string(zcomp_ultra), to_string(nseasons_ultra), year


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
parser.add_option('--outName', type=str, default='config_cosmoSN_dr_0.9.hdf5',
                  help='output file name [%default]')
parser.add_option('--outDir', type=str, default='cosmo_files',
                  help='output directory[%default]')


opts, args = parser.parse_args()

confName = opts.config.split('.')[0].split('FitParams_conf_')[-1]
confName = '_'.join(vv for vv in confName.split('_')[2:])
outDir = opts.outDir
outName = opts.outName

# check if this dir exist (and create it if necessary)
if not os.path.exists(outDir):
    os.mkdir(outDir)

search_path = '{}/*{}*.hdf5'.format(opts.fileDir, confName)
print('searching ...', search_path)
fis = glob.glob(search_path)
cosmo_scen = pd.read_csv(opts.config, delimiter=';', comment='#')


print('ggg', len(fis), cosmo_scen)

#print('looking for', outName)
# if not os.path.isfile(outName):
#    print('moving to summ')
res = make_summary(fis, cosmo_scen, runtype=opts.runtype)
# get Ny visits
Ny = int(opts.fileDir.split('_')[-3])
res['Ny'] = Ny
"""
outName = opts.config.replace('config_', '').replace(
    '.csv', '_Ny_{}.hdf5'.format(Ny))
"""
fullOut = '{}/{}.hdf5'.format(outDir, outName)
res.to_hdf(fullOut, key='cosmo')

"""
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
# plt.show()
