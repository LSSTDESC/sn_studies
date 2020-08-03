import numpy as np
from . import plt
import numpy.lib.recfunctions as rf
from scipy.interpolate import interp1d
import os

class Observations:

    def __init__(self, dbDir,dbName):

        self.dbDir = dbDir
        self.dbName = dbName

        self.tab = np.load('{}/{}.npy'.format(self.dbDir,self.dbName))
        print(self.tab.dtype)
        
    def plotSeeings(self):

       
        bands='ugrizy'

        fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(12,15))
        fig.suptitle('Simulation: {}'.format(self.dbName),fontsize=20)

        figb, axb = plt.subplots(nrows=3,ncols=2,figsize=(12,15))
        figb.suptitle('Simulation: {}'.format(self.dbName),fontsize=20)


        loc=dict(zip(bands,[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]))
        seeing = 'seeingFwhmEff'
        for band in bands:
            idx = tab['band'] == band
            iloc = loc[band][0]
            jloc = loc[band][1]
            self.plotSeeing(ax[iloc][jloc],tab[idx],band,seeing,'b',0.,iloc,jloc)
            self.plotSeeing(ax[iloc,jloc],tab[idx],band,'seeingFwhmGeom','r',0.1,iloc,jloc)
    
            self.plotSeeing(axb[iloc,jloc],tab[idx],band,seeing,'b',0.,iloc,jloc,vs_time=True)
            self.plotSeeing(axb[iloc,jloc],tab[idx],band,'seeingFwhmGeom','r',0.1,iloc,jloc,vs_time=True)
            self.seeingProba(tab[idx],band,seeing)
            self.seeingProba(tab[idx],band,'seeingFwhmGeom')
            plt.savefig('seeing_{}.png'.format(self.dbName))

    def plotSeeing(self,ax, tab,band, seeing,color,xtrans,ipos,jpos,vs_time=False):
    
        """
        Display seeing histogram
    
        Parameters:
        -----------
        ax: axis for the plot
        tab: numpy array to display
        band: band
        seeing: seeing name
        color: histogram color
        xtrans: parameter used for text display
        ipos, jpos: index of the considered acis
    
        Returns:
        -------
        None
    
    
        """
    
    
        #fig, ax = plt.subplots(nrows=1, ncols=1)
    
        #for i,val in enumerate(['seeingFwhm500','seeingFwhmGeom','seeingFwhmEff']):
        val=seeing
        med = np.median(tab[val])
        fontsize = 20
        #print(band,val,'med=',med,tab[val].min(),tab[val].max(),tab.dtype)
        #ax.hist(tab[val],histtype='step',density=True,bins=64)
        if not vs_time:
            ax.hist(tab[val],histtype='step',normed=True,bins=64,range=[0.3,2.5],linewidth=3,label=seeing)
        else:
            ax.plot(tab['mjd'],tab[val],linewidth=None,label=seeing)
        if ipos == 2:
            ax.set_xlabel('seeing ["]',fontsize=fontsize)
        if jpos == 0:
            if vs_time is False:
                ax.set_ylabel('Number of entries',fontsize=fontsize)
            else:
                ax.set_ylabel('MJD [day]',fontsize=fontsize)
            
        ax.text(0.7, 0.9, band, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,fontsize=fontsize)
        ax.text(0.7, 0.7-xtrans, 'median: '+str(np.round(med,2))+'"', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,
                fontsize=fontsize,color=color)
        ax.tick_params(labelsize = fontsize)
        ax.grid()
        if ipos==0 and jpos == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1.2),ncol=2,fontsize=fontsize,frameon=False)
    
    def seeingProba(self,tab,band,seeing):
        """
        Estimate a seeing probability distribution
        Values are saved in a .npz file: seeing_seeingname.npz
    
        Parameters:
        ----------
        tab: input numpy array
        band: band
        seeing: seeing name
    
        Returns:
        -------
        None
    
    
        """
    
        val= seeing
        print(band,val,'med=',np.median(tab[val]),tab[val].min(),tab[val].max())
        bin_width = 0.05
        min_val = tab[val].min()
        max_val = tab[val].max()
        r = self.getDistrib(tab,val,bin_width,'seeing','weight_seeing')
        fichname='seeing_{}_{}.npz'.format(band,seeing)
        np.savez(fichname, seeing = r['seeing'],weight_seeing=r['weight_seeing'])
        """
        rs=[]
        seeing_npz = np.load(fichname,'r')
        for i in range(100000):
        seeing=np.random.choice(seeing_npz['seeing'],1,p=seeing_npz['weight_seeing'])[0]
        rs.append(seeing)
        plt.hist(rs,color='k',histtype='step',density=True,bins=64)
        plt.xlim([r['seeing'].min(),r['seeing'].max()])
        """

        
    def getDistrib(self,sel,varname,bin_width,varnamea,varnameb):
        """
        Transform a distribution to a binned histogram
    
        Parameters:
        -----------
        sel: input numpy array
        varname: name of the field to consider in sel
        bin_width: width of the bin
        varnamea: name of an ouput field
        varnameb: name of an output field
    
    
        Returns:
        -------
        record array with fields varnamea and varnameb
    
        """
        r=[]                                                                                                                                                                                                                       
        #print('hello',min_val,max_val)
        min_val = sel[varname].min()
        max_val = sel[varname].max()            
        num_bins=int((max_val-min_val)/bin_width)                                                                                   
        max_val=min_val+num_bins*bin_width  
        #print('hello',min_val,max_val,num_bins)
        hista, bin_edgesa = np.histogram(sel[varname],bins=num_bins+1,range=[min_val-bin_width/2.,max_val+bin_width/2.])            
                                                                                                                                
        bin_center = (bin_edgesa[:-1] + bin_edgesa[1:]) / 2                                                                         
                                                                                                                                
        #print(min_val,max_val)                                                                                                      
        for i,val in enumerate(hista):                                                                                              
            #print(round(bin_center[i],3),float(hista[i])/float(np.sum(hista)))                                                      
            r.append((round(bin_center[i],3),float(hista[i])/float(np.sum(hista)))) 
        
        return np.rec.fromrecords(r, names=[varnamea,varnameb])

    def dist(self):

        res = {}
        cadence = 1
        for band in 'gri':
            resband = None
            idx = self.tab['band'] ==band
            sel = self.tab[idx]
            sel.sort(order='mjd')
            mjdmin, mjdmax = np.min(sel['mjd']),np.max(sel['mjd'])
            for mjd in np.arange(mjdmin, mjdmax, 365.):
        
                mjds = np.arange(mjd,mjd+365,cadence)
        
                mjd_tile = np.tile(sel['mjd'],(len(mjds),1))
                m5_tile = np.tile(sel['fiveSigmaDepth'],(len(mjds),1))
                seeingFwhmEff_tile =  np.tile(sel['seeingFwhmEff'],(len(mjds),1))
                seeingFwhmGeom_tile = np.tile(sel['seeingFwhmGeom'],(len(mjds),1))
                diff = mjd_tile-mjds[:,None]
                #print(diff)
                idxb = np.abs(diff)<1.
                flag = np.argwhere(idxb)
        
                mjd_ma = np.ma.array(mjd_tile,mask=~idxb)
                m5_ma = np.ma.array(m5_tile,mask=~idxb)
                seeingFwhmEff_ma = np.ma.array(seeingFwhmEff_tile, mask=~idxb)
                seeingFwhmGeom_ma = np.ma.array(seeingFwhmGeom_tile, mask=~idxb)
        
                mjdcol = np.ma.median(mjd_ma,axis=1)
                m5col = np.ma.median(m5_ma,axis=1)
                seeingFwhmEffcol = np.ma.median(seeingFwhmEff_ma,axis=1)
                seeingFwhmGeomcol = np.ma.median(seeingFwhmGeom_ma, axis=1)
        
                resb = np.array(mjdcol, dtype=[('mjd','f8')])
                resb= rf.append_fields(resb,'m5',m5col)
                resb= rf.append_fields(resb,'seeingFwhmEff',seeingFwhmEffcol)
                resb= rf.append_fields(resb,'seeingFwhmGeom',seeingFwhmGeomcol)
        
        
                if resband is None:
                    resband = resb
                else:
                    resband = np.concatenate((resband, resb))
            print(band,'band processed')
            res[band] = resband
            #plt.plot(res[band]['mjd'],res[band]['m5'])
            #break
    
    
        idx = self.tab['band'] == 'g'
        sel = self.tab[idx]
        sel.sort(order='mjd')
        mjdmin, mjdmax = np.min(sel['mjd']),np.max(sel['mjd'])
        mjds = np.arange(mjdmin,mjdmax,cadence)

        resfi = None
        for band in 'gri':
            tabb = res[band]
            resfib = None
            for val in ['m5','seeingFwhmEff','seeingFwhmGeom']:
                interpol = interp1d(tabb['mjd'],tabb[val],fill_value=-1,bounds_error=False)
                if resfib is None:
                    resfib = np.array(interpol(mjds), dtype=[(val,'f8')])
                else:
                    resfib = rf.append_fields(resfib,val,interpol(mjds))
            resfib = rf.append_fields(resfib,'band',[band]*len(resfib))
            resfib = rf.append_fields(resfib,'mjd',mjds)
            if resfi is None:
                resfi = resfib
            else:
                resfi = np.concatenate((resfi,resfib))
        
        np.save('distrib.npy',np.copy(resfi))


    def make_all_obs(self,nexp_expt = [(1,5),(1,15),(1,30)]):
        
        if not os.path.exists('distrib.npy'):
            self.dist()

        restot = np.load('distrib.npy')

        for (nexp,expt) in nexp_expt:
            obs = self.make_obs(restot,nexp,expt)
            namesim = 'Observations_{}_{}.npy'.format(nexp,expt)
            np.save(namesim,np.copy(obs))
            
        
    def make_obs(self,restot,nexp,exptime):
    
        mjdCol = 'observationStartMJD'
        RaCol = 'fieldRA'
        DecCol = 'fieldDec'
        filterCol = 'filter'
        m5Col = 'fiveSigmaDepth'
        exptimeCol = 'visitExposureTime'
        nightCol = 'night'
        nexpCol = 'numExposures'
        seasoncol = 'season'
        seeinga = 'seeingFwhmEff'
        seeingb = 'seeingFwhmGeom'
    

        Ra = 0.0
        Dec = 0.0

        resu = None
        for band in np.unique(restot['band']):
            idd = (restot['band'] == band)&(restot['m5']>10)
            sel = restot[idd]
    
            #plt.plot(sel['mjd'],sel['m5'])
            cad = sel['mjd'][1:]-sel['mjd'][:-1]
            print(band,np.mean(cad))
            res = np.array(sel['mjd'],dtype=[(mjdCol,'f8')])
            m5 = sel['m5']+1.25*np.log10(exptime/30.)
            res = rf.append_fields(res,m5Col,m5)
            res = rf.append_fields(res,filterCol,sel['band'])
            res = rf.append_fields(res,seeinga,sel[seeinga])
            res = rf.append_fields(res,seeingb,sel[seeingb])
    
            res = rf.append_fields(res,RaCol,[Ra]*len(res))
            res = rf.append_fields(res,DecCol,[Dec]*len(res))
            res = rf.append_fields(res,exptimeCol,[exptime]*len(res))
            res = rf.append_fields(res,nexpCol,[nexp]*len(res))

            mjdmin = np.min(sel['mjd'])
            night = sel['mjd']-mjdmin
            season = np.rint(night/365.)+1
        
            res = rf.append_fields(res,nightCol,night)
            res = rf.append_fields(res,seasoncol,season)
            if resu is None:
                resu = res
            else:
                resu = np.concatenate((resu,res))
        return resu


def prepareYaml(input_file, nexp, exptime, x1, color,season,nproc,output_file):
    """"
    Function to generate a yaml file for simulation

    """

    
    with open(input_file, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('nexp', str(nexp))
    filedata = filedata.replace('exptime', str(exptime))
    filedata = filedata.replace('x1v', str(x1))
    filedata = filedata.replace('colorv', str(color))
    filedata = filedata.replace('seasonval', str(season))
    filedata = filedata.replace('nprocval', str(nproc))

    with open(output_file, 'w') as file:
        file.write(filedata)
