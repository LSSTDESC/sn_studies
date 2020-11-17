from sn_tools.sn_io import loopStack
import glob
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import pandas as pd
from astropy.table import Table
import sncosmo
from sn_tools.sn_telescope import Telescope
import astropy.units as u


def plotLC(table, time_display=10):
    """ Light curve plot using sncosmo methods

    Parameters
    ---------------
    table: astropy table
     table with LS informations (flux, ...)
   time_display: float
     duration of the window display
    """

    telescope = Telescope(airmass=1.2)
    for band in 'grizy':
        if telescope.airmass > 0:
            band = sncosmo.Bandpass(
                telescope.atmosphere[band].wavelen, telescope.atmosphere[band].sb, name='LSST::'+band, wave_unit=u.nm)
        else:
            band = sncosmo.Bandpass(
                telescope.system[band].wavelen, telescope.system[band].sb, name='LSST::'+band, wave_unit=u.nm)
        sncosmo.registry.register(band, force=True)

    if 'x1' in table.meta.keys():
        x1 = table.meta['x1']
        color = table.meta['color']
        x0 = table.meta['x0']
    else:
        x1 = 0.
        color = 0.
        x0 = 0.
    daymax = table.meta['daymax']
    table = table[table['flux']/table['fluxerr'] >= 1.]
    print('plotting')
    model = sncosmo.Model('salt2')
    model.set(z=table.meta['z'],
              c=color,
              t0=daymax,
              # x0=x0,
              x1=x1)

    sncosmo.plot_lc(data=table)

    plt.draw()
    plt.pause(time_display)
    plt.close()


def passed(nlc_bef, nlc_aft, nbands_lc, nbef, naft, nbands):

    res = nlc_bef >= nbef
    res &= nlc_aft >= naft
    res &= nbands_lc >= nbands

    return res


def plotMollview(data, varName='nSN', leg='Number of SN', op=sum, xmin=0., xmax=12., nside=64):
    """
    Method to display results as a Mollweid map

    Parameters
    ---------------
    data: astropytable
    data to consider
    varName: str
    name of the variable to display
    leg: str
    legend of the plot
    op: operator
    operator to apply to the pixelize data(median, sum, ...)
    xmin: float
    min value for the display
    xmax: float
    max value for the display

    """
    npix = hp.nside2npix(nside)

    hpxmap = np.zeros(npix, dtype=np.float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[data['healpixID'].astype(
        int)] += data[varName]

    norm = plt.cm.colors.Normalize(xmin, xmax)
    cmap = plt.cm.jet
    cmap.set_under('w')
    resleg = op(data[varName])
    if 'nsn' in varName:
        resleg = int(resleg)
    else:
        resleg = np.round(resleg, 2)
    title = '{}: {}'.format(leg, resleg)

    hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                title=title, nest=True, norm=norm)
    hp.graticule()


def load(dbDir, dbName, fieldName, runType, inum, prefix):

    fis = get_files(dbDir, dbName, fieldName, runType, inum, prefix)
    res = loopStack(fis, objtype='astropyTable')

    return res.to_pandas()


def get_files(dbDir, dbName, fieldName, runType, inum, prefix):

    path = '{}/{}/{}*{}*{}*{}*.hdf5'.format(dbDir,
                                            dbName, prefix, fieldName, runType, inum)

    # print(path)
    fis = glob.glob(path)
    # print(fis)
    return fis


def nSN_pixels(res):

    r = []
    for healpixID in np.unique(res['healpixID']):
        idx = res['healpixID'] == healpixID
        sela = res[idx]
        for season in np.unique(sela['season']):
            io = sela['season'] == season
            selb = sela[io]
            print(healpixID, season, len(selb))
            r.append((healpixID, season, len(selb)))

    # res = np.rec.fromrecords(r, names=['healpixID', 'season', 'nSN'])

    return pd.DataFrame(r, columns=['healpixID', 'season', 'nSN'])


def plot_nSN(respix):

    xmin = np.min(respix['nSN'])
    xmax = np.max(respix['nSN'])
    plotMollview(respix, xmin=xmin, xmax=xmax)

    fig, ax = plt.subplots()
    for healpixID in np.unique(respix['healpixID']):
        idx = respix['healpixID'] == healpixID
        sela = respix[idx]
        ax.plot(sela['season'], sela['nSN'])


def plotHist(tab, var):

    fig, ax = plt.subplots()
    ax.hist(tab[var], histtype='step', bins=20)


def select(lc, daymax, snrmin=1.):
    idx = lc['flux'] > 0.
    idx &= lc['fluxerr'] > 0.

    select = lc[idx]
    select['snr'] = select['flux']/select['fluxerr']
    idx = select['snr'] >= snrmin
    select = select[idx]
    select['diff_time'] = daymax-select['time']
    nlc_bef = len(select[select['diff_time'] >= 0])
    nlc_aft = len(select[select['diff_time'] < 0])
    # check the total number of LC points here
    assert((nlc_bef+nlc_aft) == len(select))

    nbands = 0
    if len(select) > 0:
        nbands = len(np.unique(select['band']))

    return nlc_bef, nlc_aft, nbands
    """
    selb = Table()
    for b in np.unique(select['band']):
        io = select['band']==b
        selo = select[io]
        ibo = selo['snr']>=5.
        # print(b,len(selo[ibo]))
        if len(selo[ibo]) >= 2.:
            selb = vstack([selb,selo])
    

    # print(len(selb),len(select),nlc_bef,nlc_aft)
    select = Table(selb)
    """


dbDir = '/media/philippe/LSSTStorage/DD'
dirSimu = '{}/Simu'.format(dbDir)
dirFit = '{}_new/Fit'.format(dbDir)
#dirFit = 'OutputFit'
dbName = 'descddf_v1.5_10yrs'
fieldName = 'COSMOS'
runType = 'allSN'

"""


res = load(dbDir, dbName, fieldName, runType, '*', 'Simu')
print(len(np.unique(res['healpixID'])))

plotHist(res, var='z')
plotHist(res, var='x1')
plotHist(res, var='color')

plt.show()
"""
"""
print(res.columns)

respix = nSN_pixels(res)
plot_nSN(respix)
plt.show()
"""
n_SN = 0

for io in range(1, 8):
    res = load(dirSimu, dbName, fieldName, runType, io, 'Simu')
    # print(res.columns)
    lcName = get_files(dirSimu, dbName, fieldName, runType, io, 'LC')[0]
    fitName = get_files(dirFit, dbName, fieldName, runType, io, 'Fit')
    #print('oo', lcName, len(res))
    res['index_hdf5'] = res['index_hdf5'].str.decode('utf-8')
    fitlc = loopStack(fitName, objtype='astropyTable')
    # print(len(fitlc))
    assert(len(fitlc) == len(res))
    for ib, row in res.iterrows():
        key = row['index_hdf5']
        lc = Table.read(lcName, path='lc_{}'.format(key))
        #print(lc.columns, lc['zpsys'])
        #lc['zpsys'] = lc['zpsys'].str.decode('utf-8')
        lc.convert_bytestring_to_unicode()
        #print(lc.meta, len(lc))
        nlc_bef, nlc_aft, nbands = select(lc, lc.meta['daymax'], snrmin=1.)
        if passed(nlc_bef, nlc_aft, nbands, 4, 10, 3):
            # print('passed')
            n_SN += 1
        # plotLC(lc)
        # break
    break
print('number of SN', n_SN)
