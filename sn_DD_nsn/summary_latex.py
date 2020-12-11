import numpy as np


def min_med_max(val):

    return np.min(val), np.median(val), np.max(val)


def print_resu(fia, fieldName, selb, npixa, npixb):

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

    finalres += '& {} & {} & {} & {} & {} & {} & {} & {} \\\\'.format(fieldName,
                                                                      cadence, filter_alloc, m5, nights, season_length, int(npixa), int(npixb))

    # finalres += '& {} & {} & {} & {} & {}  \\\\'.format(fieldName,
    #                                                    cadence, filter_alloc, m5, nights)

    finalresb += '& {} & {} & {} & {} \\\\'.format(fieldName,
                                                   season_length, int(npixa), int(npixb))
    fia.write('{} \n'.format(finalres))
    #fib.write('{} \n'.format(finalresb))


def print_bandeau(fia):

    fia.write('\\begin{center} \n')
    fia.write('\\begin{sidewaystable}[htbp] \n')
    fia.write('\\resizebox{\\textwidth}{!}{% \n')
    fia.write('\\begin{tabular}{ccccccccc} \n ')
    fia.write(
        'Observing & Field & cadence & Nvisits & m5 & Nnights & season length & total area & effective area \\\\ \n')
    fia.write(
        ' Strategy & & min/med/max & g/r/i/z/y & g/r/i/z/y & & [days] & [deg2] & [deg2] \\\\ \n')


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

    fia.write('\end{tabular}} \n')
    fia.write('\end{sidewaystable} \n')
    fia.write('\end{center}')


def print_end_old(fia, fib):

    fia.write('\end{tabular} \n')
    fia.write('\end{table} \n')

    fib.write('\end{tabular} \n')
    fib.write('\end{table} \n')


dbNames = ['descddf_v1.5_10yrs', 'agnddf_v1.5_10yrs',
           'baseline_v1.5_10yrs', 'daily_ddf_v1.5_10yrs',
           'ddf_heavy_v1.6_10yrs', 'ddf_heavy_nexp2_v1.6_10yrs',
           'dm_heavy_v1.6_10yrs', 'dm_heavy_nexp2_v1.6_10yrs']

nfiles = int(np.round(len(dbNames)/4))

print(nfiles)

fia = open('dd_summary_1.tex', 'w')
fib = open('dd_summary_2.tex', 'w')

fia = {}
for i in range(nfiles):
    fia[i] = open('dd_summary_{}.tex'.format(i), 'w')
    print_bandeau(fia[i])

#print_bandeau(fia, fib)
idb = 0
icount = 0
for dbName in dbNames:
    icount += 1
    if icount > 4:
        icount = 0
        idb += 1

    fia[idb].write('\hline \n')
    #fib.write('\hline \n')

    fullName = 'DD_Summary_{}.npy'.format(dbName)
    res = np.load(fullName, allow_pickle=True)
    for fieldName in np.unique(res['fieldName']):
        idx = res['fieldName'] == fieldName
        sel = res[idx]
        npixels_total = len(np.unique(sel['healpixID']))
        idxb = sel['season_length'] >= 100.
        idxb = sel['nights'] >= 5
        selb = sel[idxb]
        npixels_eff = len(np.unique(selb['healpixID']))
        print_resu(fia[idb], fieldName, selb, npixels_total, npixels_eff)


for i in range(nfiles):
    print_end(fia[i])
    fia[i].close()
