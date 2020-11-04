import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd


def anafich(tab):
    idx = tab['Nvisits'] < 200.
    idx &= tab['sigmaC'] >= 0.0390
    sel = pd.DataFrame(tab[idx].copy())
    if len(sel) <= 0:
        return None

    # print(sel.columns)

    lliss = ['sigmaC', 'SNRcalc_tot', 'Nvisits']
    for b in 'grizy':
        lliss += ['Nvisits_{}'.format(b)]
        lliss += ['SNRcalc_{}'.format(b)]

    return sel[lliss]


def loopAna(fis, j=0, output_q=None):

    snr = None
    for fi in fis:
        tab = np.load(fi, allow_pickle=True)
        sel = anafich(tab)
        if sel is not None:
            if snr is None:
                snr = sel
            else:
                # print(snr,tab)
                #snr = np.concatenate((snr,sel))
                snr = pd.concat((snr, sel))

    if output_q is not None:
        return output_q.put({j: snr})
    else:
        return snr


def multiAna(thedir, nproc=8):

    fis = glob.glob('{}/*.npy'.format(thedir))
    nz = len(fis)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    #print('multi', nz, t)
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=loopAna,
                                     args=(fis[t[j]:t[j+1]], j, result_queue))
             for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = None

    # gather the results
    for key, vals in resultdict.items():
        if vals is not None:
            print(type(vals))
            if restot is None:
                restot = vals
            else:
                #restot = np.concatenate((restot, vals))
                restot = pd.concat((restot, vals))
    return restot


z = 0.8
tag = 'z_{}'.format(z)
thedir = 'SNR_combi/{}'.format(tag)

sumname = 'summary_{}.npy'.format(tag)

if not os.path.isfile(sumname):
    snr = multiAna(thedir)
    print('type', type(snr))
    np.save(sumname, snr.to_records(index=False))

snr = pd.DataFrame(np.load(sumname, allow_pickle=True))

print(snr.columns)
"""
plt.plot(snr['Nvisits'],snr['sigmaC'],'r*')
plt.show()
"""

fluxFrac = np.load('fracSignalBand.npy', allow_pickle=True)

print(fluxFrac.dtype)
idb = np.abs(fluxFrac['z']-z) < 1.e-8
fluxFrac = fluxFrac[idb]


print(fluxFrac)

bbands = 'grizy'
for b in bbands:
    io = fluxFrac['band'] == b
    flux = fluxFrac[io]
    print('flux', flux.dtype)
    snr['chisq_{}'.format(b)] = snr['SNRcalc_{}'.format(b)] / \
        snr['SNRcalc_tot']
    snr['chisq_{}'.format(b)] -= flux['fracfluxband']
    #plt.plot(snr['Nvisits_{}'.format(b)], snr['chiq_{}'.format(b)])

# plt.show()

snr['chisq'] = 0.
for b in bbands:
    snr['chisq'] += snr['chisq_{}'.format(b)] * snr['chisq_{}'.format(b)]

snr['chisq'] = np.sqrt(snr['chisq'])

snr = snr.sort_values(by=['chisq'])

#ij = snr['chisq'] <= 0.5
ij = snr['Nvisits_y'] <= 25.
ij &= snr['Nvisits_i'] >= snr['Nvisits_r']
ij &= snr['Nvisits_z'] >= snr['Nvisits_i']
snr = snr[ij]

idxa = np.argmin(snr['chisq'])
print(snr.iloc[idxa])

# print(snr[:20])
plt.plot(snr['chisq'], snr['Nvisits'], 'ko')
plt.show()
idxa = np.argmin(sel_snr['Nvisits'])
print(sel_snr.iloc[idxa])
print(snr)


idx = snr['Nvisits_y'] >= 5
idx &= snr['Nvisits_y'] <= 20
idx &= snr['Nvisits_i'] >= 10
idx &= snr['Nvisits_i'] >= snr['Nvisits_r']
idx &= snr['Nvisits_z'] >= snr['Nvisits_i']
sel_snr = snr[idx]


# plt.plot(sel_snr['Nvisits'],sel_snr['sigmaC'],'ko',mfc='None')
