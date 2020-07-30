from sn_tools.sn_telescope import Telescope
import matplotlib.pyplot as plt
import numpy as np
import numpy.lib.recfunctions as rf

def magToflux():

    telescope = Telescope(airmass=1.2,aerosol=False)

    #print(plt.rcParams.keys())
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['lines.linewidth'] = 2.
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['legend.facecolor'] = 'w'
    plt.rcParams['figure.figsize'] = (10, 7)

    #telescope.Plot_Throughputs()
    #plt.savefig('LSST_throughput.png')

    mag = np.arange(14,20.1,0.1)
    exptime=15.
    nexp = 1
    bands = 'gri'
    restot=None
    filtercolors = {'u': 'b', 'g': 'b',
                    'r': 'g', 'i': 'r', 'z': 'r', 'y': 'm'}
    for band in bands:
        flux_e_sec = telescope.mag_to_flux_e_sec(mag,[band]*len(mag),[exptime]*len(mag),[nexp]*len(mag))[:,1]
        #print(band,flux_e_sec)
        res = np.array(flux_e_sec,dtype=[('flux_e_sec','f8')])
        res = rf.append_fields(res,'band',[band]*len(res))
        res = rf.append_fields(res,'mag',mag)
        if restot is None:
            restot = res
        else:
            restot = np.concatenate((restot,res))

    #plt.ticklabel_format(style='scientific', axis='y',useMathText=True) 
    plt.gca().get_yaxis().get_major_formatter().set_powerlimits((0, 0))
    for band in bands:
        idx = restot['band']==band
        sel = restot[idx]
        plt.semilogy(sel['mag'],sel['flux_e_sec'],color=filtercolors[band],label='{} band'.format(band))
       
        #plt.ticklabel_format(style='scientific', axis='y',useMathText=True)
    plt.xlabel('mag')
    plt.ylabel('flux [pe/s]')
    plt.xlim([14.,20.])
    plt.ylim([1e3,1e6])
    plt.legend() 
    plt.grid()
    plt.savefig('flux_mag.png')
