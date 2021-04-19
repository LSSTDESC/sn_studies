import numpy as np
import healpy as hp
from optparse import OptionParser, OptionGroup
import pandas as pd


def min_med_max(val):
    """
    Function to extract min, median, max of val

    Parameters
    --------------
    val: numpy array col

    Returns
    -----------
    min, median, max

    """
    return np.min(val), np.median(val), np.max(val)


def print_resu(fia, fieldName, selb, npixa, npixb, frac_DD):
    """
    Function to print in a file some cadence results related to a field

    Parameters
    --------------
    fia: pointer
       file where the results will be written
    fieldName: str
       field to consider
    selb: numpy array
      array containing data to process
    npixa: int
      number of pixels with data for this field
    npixb: int
      number of pixels with usefull data in this field
    frac_DD: float
       DD frac for this observing strategy

    """
    cadmi, cadmed, cadmax = min_med_max(selb['cadence'])
    cadence = '{}/{}/{}'.format(
        int(cadmi), int(cadmed), int(cadmax))
    nights_mi, nights_med, nights_max = min_med_max(selb['nights'])
    nights = '{}/{}/{}'.format(
        int(nights_mi), int(nights_med), int(nights_max))
    seasmi, seasmed, seasmax = min_med_max(selb['season_length'])
    season_length = '{}/{}/{}'.format(
        int(seasmi), int(seasmed), int(seasmax))
    filter_alloc = ''
    m5 = ''
    for band in 'grizy':
        nvisits = np.median(selb['N_{}'.format(band)])
        m5b = np.median(selb['m5_{}'.format(band)])
        filter_alloc += '{}'.format(int(nvisits))
        m5 += '{}'.format(np.round(m5b, 1))
        if band != 'y':
            filter_alloc += '/'
            m5 += '/'
    finalres = ''
    finalresb = ''
    if fieldName == 'CDFS':
        finalres = dbName.replace('_', '\\_')
        finalresb = dbName
        finalres += '& {}'.format(frac_DD)
    else:
        finalres += '& '

    # finalres += '& {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(fieldName,
    #                                                                  cadence, filter_alloc, m5, nights, season_length, int(npixa), int(npixb))
    finalres += '& {} & {} & {} & {} & {} & {} \\\\'.format(fieldName,
                                                            cadence, filter_alloc, nights, season_length, int(npixa))
    # finalres += '& {} & {} & {} & {} & {}  \\\\'.format(fieldName,
    #                                                    cadence, filter_alloc, m5, nights)

    # finalresb += '& {} & {} & {} & {} \\\\'.format(fieldName,
    #                                               season_length, int(npixa), int(npixb))
    fia.write('{} \n'.format(finalres))
    # fib.write('{} \n'.format(finalresb))


def print_resu_summary(fia, dbName, frac_DD, cad_tot, nvisits_tot, nights_tot, seasonlength_tot, total_area):
    cad = '/'.join(map(str, cad_tot))
    nvisits = '/'.join(map(str, nvisits_tot))
    nights = '/'.join(map(str, nights_tot))
    seasonlength = '/'.join(map(str, seasonlength_tot))
    area = np.round(total_area, 1)
    """
    finalres = '{} & {} & {} & {} & {} & {} & {} \\\\'.format(
        dbName.replace('_', '\_'), cad, nvisits, nights, seasonlength, area, frac_DD)
    fia.write('{} \n'.format(finalres))
    """
    finalres = '{} & {} & {} & {} & {} & {} \\\\'.format(
        dbName.replace('_', '\_'), cad, nvisits, seasonlength, area, frac_DD)
    fia.write('{} \n'.format(finalres))


def print_bandeau(fia, fieldlist):
    """
    Function to write necessary latex info for tables

    Parameters
    ---------------
    fia: pointer
      file where infos will be written.

    """
    fia.write('\\begin{table}[!htbp] \n')
    fia.write('\\caption{Survey parameters for the list of observing strategies analyzed in this paper. For the cadence and season length, the numbers correspond to ' +
              fieldlist + ' fields, respectivelly.}\\label{tab:os} \n')
    # fia.write('\\begin{center} \n')
    # fia.write('\\centering')
    # fia.write('\\begin{sidewaystable}[htbp] \n')
    # fia.write('\\resizebox{1.1\\textwidth}{!}{% \n')
    fia.write('\\begin{adjustbox}{width=1.2\\linewidth,center} \n')
    """
    fia.write('\\begin{tabular}{c|c|c|c|c|c|c|c|c|c} \n ')
    fia.write(
        'Observing & DD frac(\%) & Field & cadence & Nvisits & m5 & Nnights & season length & total area & effective area \\\\ \n')
    fia.write(
        ' Strategy &  &  & min/med/max & g/r/i/z/y & g/r/i/z/y & & [days] & [deg2] & [deg2] \\\\ \n')
    """
    """
    fia.write('\\begin{tabular}{c|c|c|c|c|c|c|c|c|c} \n ')
    fia.write(
        'Observing & DD budget(\%) & Field & cadence & Nvisits & season length & area \\\\ \n')
    fia.write(
        ' Strategy &  &  & min/med/max & g/r/i/z/y & [days] & [deg2] \\\\ \n')
    """
    fia.write('\\begin{tabular}{c|c|c|c|c|c} \n ')
    fia.write(
        'Observing & cadence & Nvisits & season length & area & DD budget\\\\ \n')
    fia.write(
        ' Strategy & [days] & u/g/r/i/z/y & [days] & [deg2] &(\%)\\\\ \n')


def print_bandeau_old(fia, fib):

    fia.write('\\begin{table}[htbp] \n')
    fia.write('\\begin{tabular}{cccccc} \n ')
    fia.write(
        'Observing Strategy & Field & cadence & Nvisits & m5 & Nnights \\\\ \n')
    fia.write(
        ' & & min/med/max & g/r/i/z/y & g/r/i/z/y & \\\\ \n')

    fib.write('\\begin{table}[htbp] \n')
    fib.write('\\begin{tabular}{ccccc} \n ')
    fib.write(
        'Observing Strategy & Field & season length & total area & effective area \\\\ \n')
    fib.write(
        ' & & [days] & [deg2] & [deg2] \\\\ \n')


def print_end(fia):
    """
    Function to write necessary latex info for table closing

    Parameters
    ---------------
    fia: pointer
      file where infos will be written.

    """
    fia.write('\\end{tabular} \n')
    fia.write('\\end{adjustbox} \n')
    # fia.write('\end{sidewaystable} \n')
    # fia.write('\\end{center}')
    fia.write('\\end{table} \n')


def print_end_old(fia, fib):

    fia.write('\end{tabular} \n')
    fia.write('\end{table} \n')

    fib.write('\end{tabular} \n')
    fib.write('\end{table} \n')


parser = OptionParser()
parser.add_option("--dbList", type="str", default='List.csv',
                  help="db name [%default]")

group = OptionGroup(parser, "Warning",
                    "To run correctly this script request a file called Nvisits.npy "
                    "This file contains general infos about DD."
                    "It should have been produced with the script run_scripts/metrics/estimate_DDFrac.py")

# group.add_option("-g", action="store_true", help="Group option.")

parser.add_option_group(group)

opts, args = parser.parse_args()

toprocess = pd.read_csv(opts.dbList, comment='#')

# dbNames of interest here
dbNames = toprocess['dbName'].to_list()
"""
dbNames = ['descddf_v1.5_10yrs', 'agnddf_v1.5_10yrs',
           'baseline_v1.5_10yrs', 'daily_ddf_v1.5_10yrs',
           'ddf_heavy_v1.6_10yrs',
           'dm_heavy_v1.6_10yrs',
           'ddf_dither0.00_v1.7_10yrs',
           'ddf_dither0.05_v1.7_10yrs',
           'ddf_dither0.10_v1.7_10yrs',
           'ddf_dither0.30_v1.7_10yrs',
           'ddf_dither0.70_v1.7_10yrs',
           'ddf_dither1.00_v1.7_10yrs',
           'ddf_dither1.50_v1.7_10yrs',
           'ddf_dither2.00_v1.7_10yrs']
"""
nfiles = int(np.round(len(dbNames)/4))

print(nfiles)

fia = {}
fieldlist = []
for dbName in dbNames:
    fullName = 'DD_Summary_{}_128.npy'.format(dbName)
    res = np.load(fullName, allow_pickle=True)
    res.sort(order='fieldName')
    for fieldName in np.unique(res['fieldName']):
        fieldlist.append(fieldName)
    break

for i in range(nfiles):
    fia[i] = open('dd_summary_{}.tex'.format(i), 'w')
    print_bandeau(fia[i], '/'.join(fieldlist))

# load generic infos for DD
DD_gen = np.load('Nvisits.npy', allow_pickle=True)

# print_bandeau(fia, fib)
idb = 0
icount = 0
for dbName in dbNames:
    # get ddfrac
    iio = DD_gen['cadence'] == dbName
    frac_DD = 100.*DD_gen[iio]['frac_DD'].item()
    frac_DD = np.round(frac_DD, 1)

    icount += 1
    if icount > 20:
        icount = 0
        idb += 1

    fia[idb].write('\hline \n')
    # fib.write('\hline \n')

    nside = 128
    pixArea = hp.nside2pixarea(nside, degrees=True)
    fullName = 'DD_Summary_{}_128.npy'.format(dbName)
    res = np.load(fullName, allow_pickle=True)
    total_area = 0.
    cad_tot = []
    nvisits_tot = []
    nights_tot = []
    seasonlength_tot = []

    # for i, fieldName in enumerate(np.unique(res['fieldName'])):
    for i, fieldName in enumerate(fieldlist):
        idx = res['fieldName'] == fieldName
        sel = res[idx]
        npixels_total = len(np.unique(sel['healpixID']))
        idxb = sel['season_length'] >= 100.
        idxb = sel['nights'] >= 5
        selb = sel[idxb]
        npixels_eff = len(np.unique(selb['healpixID']))
        """
        print_resu(fia[idb], fieldName, selb,
                   npixels_total*pixArea, npixels_eff*pixArea, frac_DD)
        """
        cadmi, cadmed, cadmax = min_med_max(selb['cadence'])
        cad_tot.append(cadmed)
        nights_mi, nights_med, nights_max = min_med_max(selb['nights'])
        nights_tot.append(int(nights_med))
        seasmi, seasmed, seasmax = min_med_max(selb['season_length'])
        seasonlength_tot.append(int(seasmed))
        if i == 0:
            for band in 'grizy':
                nvisits_tot.append(int(np.median(selb['N_{}'.format(band)])))
        total_area += npixels_total*pixArea
    print(dbName, np.round(total_area, 1), cad_tot, nvisits_tot)
    print_resu_summary(fia[idb], dbName, frac_DD, cad_tot, nvisits_tot, nights_tot,
                       seasonlength_tot, total_area)
for i in range(nfiles):
    print_end(fia[i])
    fia[i].close()
