import numpy as np
from . import plt


class Observations:

    def __init__(self, dbDir,dbName):

        self.dbDir = dbDir
        self.dbName = dbName

    def plotSeeings(self):

        tab = np.load('{}/{}.npy'.format(self.dbDir,self.dbName))
        print(tab.dtype)
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
