import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

def multiLoad(thedir, z,nproc=8):
    """
    Function to load and process data - multiprocessing
    
    Parameters
    --------------
    thedir: str
        location directory of the files to produce
    nproc: int, opt
        number of procs to use (default: 8)

    """
    search_path = '{}/z_{}/*.npy'.format(thedir,np.round(z,2))
    print(search_path)
    fis = glob.glob(search_path)
    nz = len(fis)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=loopLoad,
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

    restot = pd.DataFrame()

    # gather the results
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals))
    return restot

def loopLoad(fis, j=0, output_q=None):
    """
    Function to load and process data
    
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

    snr = pd.DataFrame()
    for fi in fis:
        #print('loading', fi)
        tab = pd.DataFrame(np.load(fi, allow_pickle=True))
        sel = anafich(tab)
        snr = pd.concat((snr, sel))

    if output_q is not None:
        return output_q.put({j: snr})
    else:
        return snr
    
def anafich(tab):
    """
    Function to analyze and select a data set
    
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
    #idx &= tab['sigmaC'] >= 0.0
    sel = pd.DataFrame(tab[idx].copy())
    if len(sel) <= 0:
        return pd.DataFrame()

    return sel
    
theDir = 'dd_design_faint/SNR_combi'
zref = 0.7

res = multiLoad(theDir,zref)
idx = res['Nvisits_r']<= 2
idx &= res['Nvisits_g']<= 1
res = res[idx]
res = res.sort_values(by=['Nvisits'])

nout = np.min([len(res), 10])
colout = ['sigmaC', 'SNRcalc_tot', 'Nvisits', 'Nvisits_g', 'SNRcalc_g',
                       'Nvisits_r', 'SNRcalc_r', 'Nvisits_i', 'SNRcalc_i', 'Nvisits_z',
                       'SNRcalc_z', 'Nvisits_y', 'SNRcalc_y']
colout = ['sigmaC','Nvisits', 'Nvisits_g', 'Nvisits_r', 'Nvisits_i', 'Nvisits_z','Nvisits_y',]

print(res[:nout][colout])

#print(res.columns)

sel = res
plt.plot(sel['Nvisits'],sel['sigmaC'],'ko')
plt.show()
