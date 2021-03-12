import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sn_tools.sn_telescope import Telescope
import os
from sn_tools.sn_obs import season
from optparse import OptionParser


def process(fileDir, dbName, nside, fieldName, outName):
    """
    Function to process an obspixel file and to perform coadds per night

    Parameters
    --------------
    fileDir: str
      location dir of the files
    dbName: str
      OS name
    fieldName: str
      field to consider
    outName: str
       output file name (npy type)

    """
    fullName = '{}/{}/{}_DD_nside_{}*{}.npy'.format(
        fileDir, dbName, dbName, nside, fieldName)

    fi = glob.glob(fullName)
    print(fullName, len(fi))
    tab = pd.DataFrame(np.load(fi[0], allow_pickle=True))

    print(tab.columns)
    telescope = Telescope(airmass=1.2)
    df = tab.groupby(['healpixID', 'filter', 'night']).apply(
        lambda x: coadd_night(x, telescope)).reset_index()

    print(df['fiveSigmaDepth']-df['fiveSigmaDepth_precise'])

    np.save(outName, season(df.to_records(index=False)))


def coadd_night(grp, telescope=None):
    """
    Function to perform some modifs per night

    Parameters
    --------------
    grp : pandas group (here of healpixID, filter, night)

    Returns
    ----------
    pandas df with the following cols:
    observationStartMJD, fieldRA, fieldDec: mean of the night
    pixRA, pixDec, fiveSigmaDepth, seeingFwhmEff: median of the night
    visitExposureTime, numExposures: sum of the night
    fiveSigmaDepth_coadd: coadded m5 = m5_med+1.25*np.log10(Nvisits)
    """

    dictres = {}

    means = ['observationStartMJD', 'fieldRA', 'fieldDec']
    meds = ['pixRA', 'pixDec', 'fiveSigmaDepth', 'seeingFwhmEff']
    sums = ['visitExposureTime', 'numExposures']

    for vv in means:
        dictres[vv] = grp[vv].mean()

    for vv in meds:
        dictres[vv] = grp[vv].median()

    for vv in sums:
        dictres[vv] = grp[vv].sum()

    dictres['fiveSigmaDepth_coadd'] = dictres['fiveSigmaDepth']+1.25 * \
        np.log10(dictres['visitExposureTime']/30.)

    if telescope is not None:
        #sigma5_all = telescope.mag_to_flux_e_sec(grp['fiveSigmaDepth'],[grp.name[1]]*len(grp),grp['visitExposureTime'],grp['numExposures'])/5.
        sigma5_all = []
        for vv in grp['fiveSigmaDepth']:
            sigma5_all.append(telescope.mag_to_flux(vv, grp.name[1])/5.)
        sigma5_all = np.asarray(sigma5_all)
        sigma5 = 1./np.sqrt(np.sum(1./(sigma5_all*sigma5_all)))
        flux5 = 5.*sigma5

        m5 = telescope.flux_to_mag(flux5, grp.name[1])
        dictres['fiveSigmaDepth_precise'] = m5.item()

    dictout = {}
    for key, vals in dictres.items():
        dictout[key] = [vals]

    return pd.DataFrame.from_dict(dictout)


parser = OptionParser()

parser.add_option("--fileDir", type=str, default='../ObsPixelized_128',
                  help="OS dir location (pixels)[%default]")
parser.add_option("--dbName", type=str, default='descddf_v1.5_10yrs',
                  help="OS name[%default]")
parser.add_option("--fieldName", type=str, default='COSMOS',
                  help="field to consider for this study  [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="healpix nside [%default]")
parser.add_option("--outputDir", type=str, default='pixel_analysis',
                  help="output directory for data [%default]")

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName
fieldName = opts.fieldName
nside = opts.nside
outputDir = opts.outputDir

if not os.path.exists(outputDir):
    os.mkdir(outputDir)

outName = '{}/ObsPixelized_{}_{}_{}_night.npy'.format(
    outputDir, nside, dbName, fieldName)

if not os.path.isfile(outName):
    process(fileDir, dbName, nside, fieldName, outName)

"""
print('loading', outName)
#tab = pd.DataFrame(np.load(outName, allow_pickle=True))
tabo = np.load(outName, allow_pickle=True)
print(season(tabo))
tab = pd.DataFrame(np.copy(season(tabo)))
tab['medm5'] = tab.groupby(['healpixID', 'filter', 'season'])[
    'fiveSigmaDepth'].transform('mean')

for b in 'grizy':
    fig, ax = plt.subplots()
    idx = tab['filter'] == b
    sel = tab[idx]
    ax.plot(sel['night'], sel['fiveSigmaDepth_coadd'] -
            sel['fiveSigmaDepth'], 'ko')
    diff = sel['fiveSigmaDepth_coadd']-sel['medm5']
    # ax.hist(diff,histtype='step')
    # ax.plot(sel['season'],diff,'ko')
    print(b, np.mean(diff), np.std(diff))

plt.show()
"""
