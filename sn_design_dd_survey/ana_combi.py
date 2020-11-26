import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd

class CombiChoice:
    """
    class to choose SNR combination
    """
    def __init__(self):

        # get the flux fraction per band
        self.fluxFrac = np.load('fracSignalBand.npy', allow_pickle=True)

    def __call__(self,z):
        """
        Method to choose a SNR combi 

        Parameters
        --------------
        z: float
          redshift

        """
        # getting flux frac
        idb = np.abs(self.fluxFrac['z']-z) < 1.e-8
        self.fluxFrac_z = self.fluxFrac[idb]
        
        # getting the data
        tag = 'z_{}'.format(z)
        thedir = 'SNR_combi/{}'.format(tag)
        
        self.snr = self.multiAna(thedir)

        if self.snr is not None:
            self.chisq()

            print(self.snr)
            self.snr['z'] = z
            return pd.DataFrame(self.snr[:1])
        return self.snr
        
    def multiAna(self,thedir, nproc=8):
        """
        Method to load and process data - multiprocessing

        Parameters
        --------------
        thedir: str
          location directory of the files to produce
        nproc: int, opt
          number of procs to use (default: 8)

        """
        fis = glob.glob('{}/*.npy'.format(thedir))
        nz = len(fis)
        t = np.linspace(0, nz, nproc+1, dtype='int')
        #print('multi', nz, t)
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.loopAna,
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

    def loopAna(self,fis, j=0, output_q=None):
        """
        Method to load and process data

        Parameters
        --------------
        fis: list(str)
          list of files to produce
        j: int, opt
           integer for multiprocessing (proc number) (default:0)
        output_q: multiprocessing.queue
          default: None

        Returns
        ----------
        pandas df with data

        """
        
        snr = None
        for fi in fis:
            print('loading',fi)
            tab = np.load(fi, allow_pickle=True)
            # analyzing the file here
            sel = self.anafich(tab)
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

    def anafich(self,tab):
        """
        Method to analyze and select a data set

        Parameters
        ---------------
        tab: pandas df
          data to process

        Returns
        ----------
        pandas df of analyzed data

        """
        
        idx = tab['Nvisits'] < 100000000.
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

    def chisq(self):
        """
        Method to evaluate a 'chisq' wrt flux bands
        """
        
        bbands = 'grizy'
        for b in bbands:
            io = self.fluxFrac_z['band'] == b
            flux = self.fluxFrac_z[io]
            print('flux', flux.dtype)
            self.snr['chisq_{}'.format(b)] = self.snr['SNRcalc_{}'.format(b)] / \
                self.snr['SNRcalc_tot']
            self.snr['chisq_{}'.format(b)] -= flux['fracfluxband']

        self.snr['chisq'] = 0.
        for b in bbands:
            self.snr['chisq'] += self.snr['chisq_{}'.format(b)] * self.snr['chisq_{}'.format(b)]

        self.snr['chisq'] = np.sqrt(self.snr['chisq'])

        self.snr = self.snr.sort_values(by=['chisq'])
    
combi = CombiChoice()

snr_choice = pd.DataFrame()
for z in np.round(np.arange(0.3,0.9, 0.05), 2):
    z = np.round(z,2)
    res = combi(z)
    if res is not None:
        snr_choice = pd.concat((snr_choice,res))

print(snr_choice)
np.save('snr_choice.npy',snr_choice.to_records(index=False))

print(test)


plt.plot(snr['Nvisits'],snr['sigmaC'],'r*')
plt.show()

print(snr[todisp])
ij = snr['chisq'] <= 0.5
"""
ij = snr['Nvisits_y'] <= 50.
ij &= snr['Nvisits_i'] >= snr['Nvisits_r']
ij &= snr['Nvisits_z'] >= snr['Nvisits_i']
"""
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
