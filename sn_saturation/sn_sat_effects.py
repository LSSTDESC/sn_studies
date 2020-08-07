import h5py
from astropy.table import Table, vstack
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from . import plt
import multiprocessing

class LC:
    """
    class to load LC from h5py files

    """
    def __init__(self,dirFile, prodid):

        self.dirFile = dirFile
        self.prodid = prodid

        # load simulation parameters
        parName = 'Simu_{}.hdf5'.format(self.prodid)
        params = self.load_params('{}/{}'.format(self.dirFile, parName))
        # loop on this file using the simuPars list
        # select only LC with status=1
        ik = params['status'] == 1
        self.simuPars = params[ik]
        self.lcFile = '{}/LC_{}.hdf5'.format(self.dirFile, self.prodid)

    def load_params(self, paramFile):
        """
        Function to load simulation parameters

        Parameters
        ---------------
        paramFile: str
          name of the parameter file

        Returns
        -----------
        params: astropy table
        with simulation parameters

        """

        f = h5py.File(paramFile, 'r')
        print(f.keys(), len(f.keys()))
        params = Table()
        for i, key in enumerate(f.keys()):
            pars = Table.read(paramFile, path=key)
            params = vstack([params, pars])

        return params


    def getLC(self, index_hdf5):

         lc = Table.read(self.lcFile, path='lc_{}'.format(index_hdf5))

         return lc


class SaturationTime:
    """
    class to estimate saturation time vs z


    Parameters
    ---------------
    dirFile: str
      location directory wher LC are located
    x1: float
      SN x1
    color: float
      SN color
    nexp: int
      number of exposures
    exptime: float
      exposure time
   season: int
      season of observation
   cadence: int
       cadence of observation
   psf_profile: str
      psf profile (default: single_gauss)

    """
    def __init__(self, dirFile, x1,color,nexp,exptime, season,cadence,psf_profile='single_gauss'):

        prodid = 'Saturation_{}_{}_{}_{}_{}_{}_0'.format(nexp,exptime,x1,color,cadence,season)
        
        self.lc = LC(dirFile,prodid)

        pixel_npy = np.load('PSF_pixels_{}_summary.npy'.format(psf_profile))
        """
        print(pixel_npy.dtype)
        print(pixel_npy['seeing'])
        print(pixel_npy['pixel_frac_med'])
        """
        self.pixel_max = interp1d(pixel_npy['seeing'], pixel_npy['pixel_frac_med'], fill_value=0.0, bounds_error=False)

        seeings = dict(zip(['single_gauss', 'double_gauss'],
                                ['seeingFwhmEff', 'seeingFwhmGeom']))

        self.seeing_simu = seeings[psf_profile]
        #print(self.lc.simuPars)


    def multi_time(self, full_well,npp=8):

        nlc= len(self.lc.simuPars)
        batch = np.linspace(0, nlc, npp+1, dtype='int')
        
        result_queue = multiprocessing.Queue()

        for i in range(npp):

            ida = batch[i]
            idb = batch[i+1]

            p = multiprocessing.Process(name='Subprocess', target=self.time_of_saturation, args=(
                full_well, self.lc.simuPars[ida:idb], i, result_queue))
            p.start()

        resultdict = {}
        
        for j in range(npp):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        df = pd.DataFrame()
        for j in range(npp):
            df = pd.concat((df,resultdict[j]))

        return df
        

    def time_of_saturation(self, full_well,simuPars,j=0, output_q=None):
        """
        Method to estimate the time of saturation wrt full_well

        Parameters
        --------------
        full_well: float
         full well value (in pe)

        """

        dfres = pd.DataFrame()
        for par in simuPars:
            lc = self.lc.getLC(par['index_hdf5'])
            
            lc.sort('time')
           
            lc.convert_bytestring_to_unicode()

            for b in np.unique(lc['band']):
                io = lc['band'] == b
                mydf = pd.DataFrame(np.array(lc[io]))
                dg = self.time_saturation_band(mydf,full_well,lc.meta['daymax'])
                dg['x1'] =  lc.meta['x1']
                dg['color'] =  lc.meta['color']
                dg['daymax'] = lc.meta['daymax']
                dg['z'] = lc.meta['z']
                dg['band'] = b
                dfres = pd.concat((dfres,dg))
            
            #print(dfres)
            #print(test)
            """
            dfa = pd.DataFrame(np.array(lc))
            print(dfa[['band','flux_e_sec','time']])
            df = dfa.groupby(['band']).apply(lambda x: self.time_saturation_band(x,full_well)).reset_index()
            print(df)
            print(test)
            df['x1'] =  lc.meta['x1']
            df['color'] =  lc.meta['color']
            df['daymax'] = lc.meta['daymax']
            df['z'] = lc.meta['z']
            dfres = pd.concat((dfres,df))
            """
        if output_q is not None:
            output_q.put({j: dfres})
        else:
            return dfres
            

    def time_saturation_band(self, grp,full_well,T0):
        """
        Method to estimate the saturation time per band

        Parameters
        --------------
        grp: pandas group
          data to process
        full_well: float
          full well value (in pe)

        Returns
        ----------
        pandas df with the following columns:
        full_well,tBeg,tSa,tSat_interp,exptime

        """
        #print(grp[['band','flux_e_sec','visitExposureTime','time']])
        grp['flux_e'] = grp['flux_e_sec']*grp['visitExposureTime']*self.pixel_max(grp[self.seeing_simu])


        """
        print(grp.columns)
        import matplotlib.pyplot as plt
        plt.suptitle('{} band '.format(np.unique(grp['band'])))
        plt.plot(grp['phase'],grp['flux_e'])
        #plt.plot(grp['time'],grp['flux_e_sec']*grp['visitExposureTime'])
        plt.plot(grp['phase'],grp['flux_e_sec'])
        plt.show()
        """
        timeBegin = 0.
        timeSat = 0.
        timeSat_interp =0.
        tmin = np.min(grp['time'])
        tmax = np.max(grp['time'])
        ttime = np.arange(tmin,tmax,1.)

        lcinterp = interp1d(grp['time'],grp['flux_e'],fill_value=0.0, bounds_error=False)
        fluxtime = lcinterp(ttime)

        isnr = grp['snr_m5'] >= 5.
        lcsnr = grp[isnr]
        
        isat = grp['flux_e'] >= full_well
        lcsat = grp[isat]
        inosat = grp['flux_e'] <= full_well
        lcnosat = grp[inosat]
        
        
        isattime = fluxtime >= full_well
        #print('snr',lcsnr['time'])
        #print('sat',lcsat['time'])
        if len(lcsnr) > 0.:
            timeBegin = lcsnr['time'].iloc[0]
                
        if len(lcsat) > 0.:
            timeSat = lcsat['time'].iloc[0]
            timeSat_interp = ttime[isattime][0]

        idxt = np.abs(lcnosat['time']-T0) <=5.
        np_near_max = len(lcnosat[idxt])
        """
        df = pd.DataFrame([lc.meta['z']],columns=['z'])
        df['x1'] =  lc.meta['x1']
        df['color'] =  lc.meta['color']
        df['daymax'] = lc.meta['daymax']
        """
        df = pd.DataFrame([full_well],columns=['full_well'])
        df['tBeg'] = timeBegin
        df['tSat'] = timeSat
        df['tSat_interp'] = timeSat_interp
        df['exptime'] = np.mean(grp['visitExposureTime'])
        df['npts_around_max'] = np_near_max
        return df


def plotTimeSaturation(x1,color,data):
    """
    Function to plot time of saturation vs redshift
    
    Parameters
    --------------
    ax: matplotlib axis
      where the plot will be made
    data: pandas df
      data to display

    """

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    fig.subplots_adjust(right=0.75)
    
    print(data.columns)
    idx = np.abs(data['x1']-x1)<1.e-5
    idx &= np.abs(data['color']-color)<1.e-5

    df = data[idx]

    df['deltaT'] = df['tSat_interp']-df['tBeg']
    
    # get median values

    dfmed = df.groupby(['band','full_well','exptime','z'])['deltaT'].median().reset_index()

    print(dfmed)

    configs = [(90000.,15.),(90000.,30.),(120000.,15.),(120000.,30.)]
    pos = dict(zip([(0,0),(1,0),(0,1),(1,1)],configs))

    for key, vals in pos.items():
        i = key[0]
        j = key[1]
        myax = ax[i,j]
        plotBands(myax,vals[0],vals[1],dfmed)
        thetext = 'Exptime: {} s'.format(int(vals[1]))
        myax.grid()
        if i== 0 and j== 1:
            myax.legend(loc='center left', bbox_to_anchor=(
                1.1, -0.1), ncol=1)
        if i == 0:
            myax.set_title('fw = {}k pe'.format(
                int(vals[0]/1000)))
        if j== 0:
            myax.set_ylabel(r'$\Delta_{t}$ [days]')
        if j == 1:
            myax.set_yticklabels([])
            myax.text(1.1, 0.5, thetext, horizontalalignment='center', rotation=270.,
                      verticalalignment='center', transform=myax.transAxes,fontsize=20)
    # Defining custom 'xlim' and 'ylim' values.
    custom_xlim = (0.01, 0.03)
    custom_ylim = (5.,17.5)
    # Setting the values for all axes.
    plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
    
def plotBands(ax,full_well,exptime,data,bands='gri'):

    colors = dict(zip(bands, ['b', 'r', 'm']))

    idx = np.abs(data['full_well']-full_well)<1.e-5
    idx &= np.abs(data['exptime']-exptime)<1.e-5
    sel = data[idx]
    print('hhh',full_well,exptime)
    for b in bands:
        idx = sel['band']=='LSST::{}'.format(b)
        idx &= sel['deltaT']>=0.
        selb = sel[idx]
        print(selb)
        ax.plot(selb['z'],selb['deltaT'],color=colors[b],label='{} band'.format(b))
    
def gefficiency(grp):

    idx = (grp['band'] == 'LSST::g') & (grp['z'] >= 0.01)
    sel = pd.DataFrame(grp[idx])

    df = sel.groupby(['z']).apply(lambda x: calcEffi(x)).reset_index()
    
    return df
def calcEffi(grp):

    isel = grp['npts_around_max']>=3
    return pd.DataFrame({'effi': [len(grp[isel])/len(grp)]})

          
def plot_gefficiency(x1, color, tab):

    name = dict(zip([(0.0, 0.0), (-2.0, 0.2), (2.0, -0.2)],
                    ['medium', 'faint', 'bright']))

    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 15
    print(np.unique(tab[['exptime', 'full_well']]))
    """
    for exptime, fwell in np.unique(tab[['exptime', 'full_well']]):
        ida = np.abs(tab['exptime']-exptime) < 1.e-5
        ida &= (np.abs(tab['full_well']-fwell) < 1.e-5)
        snrtab = tab[ida]
        idx = (snrtab['band'] == 'LSST::g') & (snrtab['z'] >= 0.01)
        sel = Table(snrtab[idx])

        groups = sel.group_by('z')
        
        for group in groups.groups:
            z = np.unique(group['z'])[0]
            norm = len(group)
            isel = group['npts_around_max']>=3
            r.append((x1,color,z,exptime,fwell,len(group[isel])/norm,key))
        
    resu = np.rec.fromrecords(r, names=['x1','color','z','exptime','fwell','eff','psf_profile'])
    """
    resu = tab.groupby(['exptime', 'full_well']).apply(lambda x: gefficiency(x)).reset_index()
    print(resu)
    refs = [(15., 90000.), (15., 120000.), (30., 90000.), (30., 120000.)]

    ls = dict(zip(refs,['solid','dotted','dashed','dashdot']))
    colors = dict(zip(refs,['k','b','r','m']))
    ls = dict(zip(['double_gaussian','single_gaussian'],['solid','solid']))

    for exptime, fwell in refs:
        ida = np.abs(resu['exptime']-exptime) < 1.e-5
        ida &= (np.abs(resu['full_well']-fwell) < 1.e-5)
        sel = resu[ida]
      
        ax.plot(sel['z'],sel['effi'], label='({}s,{} kpe)'.format(int(exptime),int(fwell/1000)), color=colors[(exptime, fwell)])
            
    """
    ib = -1
    for key, val in ls.items():
        ib +=1 
        #ax.hlines(0.9,0.011+5.*0.01*ib,0.013+5.*0.01*ib,linestyles=val)
        ax.hlines(0.97-0.05*ib,0.010,0.011,linestyles=val)
        ax.text(0.0112,0.97-0.05*ib-0.007,corresp[key],fontsize=fontsize-3)
    """
    ax.legend(loc='lower right', fontsize=fontsize)
    ax.set_xlim([0.0095,0.05])
    ax.set_ylim([0.0,1.01])
    ax.set_xlabel('z', fontsize=fontsize)
    ax.set_ylabel('Efficiency', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid()
    plt.savefig('Plot_Sat/Effi_{}.png'.format(name[(x1, color)]))
