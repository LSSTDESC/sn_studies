import os
import pandas as pd
import numpy as np

from .signal_bands import RestFrameBands,SignalBand
from .ana_file import AnaMedValues,Anadf
from .utils import flux5_to_m5,m5_to_flux5
from sn_tools.sn_io import loopStack


class Data:
    def __init__(self,theDir,fname,
                 x1=-2.0,
                 color=0.2,
                 blue_cutoff=380.,
                 red_cutoff=800.,
                 bands='grizy'):
        """
        class to handle data: 
        - LC points
        - m5 values
        - 

        Parameters
        --------------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file
        x1: float, opt
         SN strech parameter (default: -2.0)
        color: float, opt
         SN color parameter (default: 0.2)
        blue_cutoff: float, opt
         wavelength cutoff for SN (blue part) (default: 380 nm)
        red_cutoff: float, opt
         wavelength cutoff for SN (red part) (default: 800 nm)
        bands: str, opt
         filters to consider (default: grizy)

        """


        self.x1 = x1
        self.color = color
        self.bands = bands
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff

        # load lc
        lc = self.load_data(theDir, fname)

        # estimate zcutoff by bands
        self.zband = RestFrameBands(blue_cutoff=blue_cutoff,
                         red_cutoff=red_cutoff).zband

        #apply these cutoffs
        self.lc = self.wave_cutoff(lc)
        
        # get the flux fraction per band
        self.fracSignalBand = SignalBand(self.lc)

        # load median m5

        self.m5_FieldBandSeason, self.m5_FieldBand, self.m5_Band = self.load_m5('medValues.npy')

    def load_data(self, theDir, fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file

        Returns
        -----------
        pandas df with LC point infos (flux, fluxerr, ...)
        corresponding to (x1,color) parameters  
        """

        # get LC

        lcData = self.load(theDir,fname)
         
        # select data for the type of SN (x1,color)

        idx = (lcData['x1']-self.x1)<1.e-1
        idx &= (lcData['color']-self.color)<1.e-1
        lcData = lcData[idx]

        #remove lc points with negative flux

        idx = lcData['flux_e_sec']>=0
        lcData = lcData[idx]
         
        # transform initial lcData to a pandas df
        
        lcdf = pd.DataFrame(np.copy(lcData))
        lcdf['band'] = lcdf['band'].map(lambda x: x.decode()[-1])

        return lcdf
        
    def load(self,theDir,fname):
        """
        Method to load LC data

        Parameters
        ----------
        theDir: str
          directory where the input LC file is located
        fname: str 
          name of the input LC file

        Returns
        -----------
        astropy table with LC point infos (flux, fluxerr, ...)  
        """
         
        searchname = '{}/{}'.format(theDir, fname)
        name, ext = os.path.splitext(searchname)
        
        print(searchname)
        res = loopStack([searchname], objtype='astropyTable')
        
        return res

    def wave_cutoff(self,df):
        """
        Method to select lc data (here as df)
        Selection applied:
        - bands corresponding to self.bands
        - wavelenght cutoffs from self.zband

        Parameters
        ----------
        df: pandas df with LC infos

        Returns
        -----------
        pandas df (selected)   
        """

        # select obs with bands in self.bands

        df['selband'] = df['band'].isin(list(self.bands))

        idx = df['selband'] == True

        df = df[idx]

        # select zband vals with band in self.bands

        zbanddf = pd.DataFrame(np.copy(self.zband))

        zbanddf['selband'] = zbanddf['band'].isin(list(self.bands))

        idx = zbanddf['selband'] == True

        zbanddf = zbanddf[idx]

        # now merge the two dfs

        res = df.merge(zbanddf,left_on=['band','selband'],right_on=['band','selband'])

        # select obs with z > z_blue (blue cutoff)

        idx = res['z']<= res['z_blue']
        idx &= res['z'] >= res['z_red']

        res = res[idx]

        #remove selband and z_blue and z_red
        res = res.drop(columns=['selband','z_blue','z_red'])
        
        return res

    def load_m5(self,m5_file):

        """
        Method to load fiveSigmaDepth(m5) values

        Parameters
        ----------
        m5_file: str
         m5 file name
         

        Returns
        -----------
        median_m5_field_filter_season: pandas df
         median m5 per field per filter and per season
        median_m5_field_filter: pandas df
         median m5 per field per filter
        median_m5_filter: pandas df
         median m5 per filter

        """


        self.anamed = AnaMedValues(m5_file)

        median_m5_field_filter_season = self.anamed.median_m5_field_filter_season
        median_m5_field_filter  = self.anamed.median_m5_field_filter
    
        median_m5_filter= median_m5_field_filter.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()
        #medm5_season = median_m5_season.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()

        return median_m5_field_filter_season,median_m5_field_filter,median_m5_filter


    def plotzlim(self):
        """
        Method to plot zlim

        Parameters
        ----------

        Returns
        -----------
        Two plots in one figure:
        - sigma_C vs z
        - SNR_band vs sigma_C
        
        """
        Anadf(self.lc).plotzlim()

    def plotFracFlux(self):
        """
        Method to plot fraction of flux per band vs z

        Parameters
        ----------

        Returns
        -----------
        plot of the flux fraction per band vs z
        
        """

        self.fracSignalBand.plotSignalBand(self.x1,self.color)

    def plot_medm5(self):
        """
        Method to plot m5 values

        Parameters
        ----------

        Returns
        -----------
        plot of m5 values vs field
        
        """
        self.anamed.plot(self.m5_FieldBandSeason, self.m5_FieldBand)


class Nvisits_cadence:
    def __init__(self,snr_calc,cadence,m5_type,choice_type,bands):
        

        outName = 'Nvisits_cadence_{}_{}.npy'.format(choice_type,m5_type)

        if not os.path.isfile(outName):
            self.cadence = cadence
            self.snr_calc = snr_calc

            self.cols = ['z','Nvisits']
            for band in bands:
                self.cols.append('Nvisits_{}'.format(band))

            medclass = AnaMedValues('medValues.npy')
            m5 = eval('{}.{}'.format('medclass',m5_type))
            df_tot = pd.DataFrame()

            df_tot = m5.groupby(['fieldname','season']).apply(lambda x : self.getVisits(x)).reset_index()


            self.nvisits_cadence = df_tot

            np.save(outName,np.copy(df_tot.to_records(index=False)))

        else:
            self.nvisits_cadence = pd.DataFrame(np.load(outName))


    def getVisits(self,grp):

       
        
        df = Nvisits_m5(self.snr_calc,grp).nvisits
        io = np.abs(df['cadence']-self.cadence)<1.e-5
        print('iiii',df.columns)
        df = df.loc[io,self.cols]
        #df.loc[:,'fieldname'] = grp.name[0]
        #df.loc[:,'season'] = grp.name[1]
        return df

    def plot(self):

        # this for the plot
        print(self.nvisits_cadence.groupby(['fieldname','z']).apply(lambda x: np.min(x['Nvisits'])).reset_index())

        df = self.nvisits_cadence.groupby(['fieldname','z']).agg({'Nvisits': ['min','max','median']})
        # rename columns
        df.columns = ['Nvisits_min', 'Nvisits_max', 'Nvisits_median']

        # reset index to get grouped columns back
        df = df.reset_index()

        for fieldname in df['fieldname'].unique():
            io = df['fieldname']==fieldname
            sel = df[io]
            fig, ax = plt.subplots()
            fig.suptitle(fieldname)

            ax.fill_between(sel['z'],sel['Nvisits_min'],sel['Nvisits_max'],color='grey')
            ax.plot(sel['z'],sel['Nvisits_median'],color='k')

            ax.grid()
            ax.set_xlim([0.3,0.85])
            ax.set_xlabel(r'z')
            ax.set_ylabel(r'Number of visits')
 
            figb, axb = plt.subplots()
            figb.suptitle(fieldname)

            axb.fill_between(sel['z'],sel['Nvisits_min']-sel['Nvisits_median'],sel['Nvisits_max']-sel['Nvisits_median'],color='grey')
            #axb.plot(sel['z'],sel['Nvisits_median'],color='k')

            axb.grid()
            axb.set_xlim([0.3,0.85])
            axb.set_xlabel(r'z')
            axb.set_ylabel(r'$\Delta$Number of visits')

class Nvisits_m5:
    def __init__(self, tab, med_m5):

        cols = tab.columns[tab.columns.str.startswith('m5calc')].values

        self.bands = ''.join([col.split('_')[-1] for col in cols])
        self.f5_to_m5 = flux5_to_m5(self.bands)
        self.m5_to_f5 = m5_to_flux5(self.bands)
        self.snr = tab

        self.med_m5 = self.transform(med_m5)

        print('here medians',self.med_m5)

        idx = tab['z']>=0.2
        tab = tab[idx]

        self.nvisits = self.estimateVisits()


    def estimateVisits(self,):

        dict_cad_m5 = {}
        
        cads = pd.DataFrame(np.arange(0.,10.,1.),columns=['cadence'])

        idx = 0

        #add and index to both df for the merging

        m5 = self.med_m5.copy()
        snr = self.snr.copy()

        m5.loc[:,'idx'] = idx
        snr.loc[:,'idx'] = idx

        snr = snr.merge(m5,left_on=['idx'],right_on=['idx'])
        
        zvals = snr['z'].unique()
        
        df_combi = self.make_combi(zvals,cads) 

        df_merge = snr.merge(df_combi,left_on=['z'],right_on=['z'])

        print(df_merge)
        cols = []
        for b in self.bands:
            df_merge['flux_5_e_sec_{}'.format(b)]=df_merge['flux_5_e_sec_{}'.format(b)]/np.sqrt(df_merge['cadence'])
            df_merge['m5calc_{}'.format(b)] = self.f5_to_m5[b](df_merge['flux_5_e_sec_{}'.format(b)])
            df_merge.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(df_merge['m5calc_{}'.format(b)]-df_merge['m5single_{}'.format(b)]))
            cols.append('Nvisits_{}'.format(b))
            df_merge['Nvisits_{}'.format(b)]=df_merge['Nvisits_{}'.format(b)].fillna(0.0)
           
        
        #estimate the total number of visits
        df_merge.loc[:,'Nvisits'] = df_merge[cols].sum(axis=1)
        

        return df_merge


        """
        for val in med_m5:
            for season in sel['season'].values:
                seas= s[sel['season']==season]
                tab.loc[:,'season'] = season
                test=tab.merge(seas,left_on=['season'],right_on=['season'])
            tab.loc[:,'season'] = val['season']
        """

        for fieldname in med_m5['fieldname'].unique():
            idx = med_m5['fieldname']==fieldname
            sel = med_m5[idx]
            for season in sel['season'].values:
                seas= sel[sel['season']==season]
                tab.loc[:,'season'] = season
                test=tab.merge(seas,left_on=['season'],right_on=['season'])
                
                zvals = test['z'].unique()

                df_combi = self.make_combi(zvals,cads)

                #merge with test

                df_merge = test.merge(df_combi,left_on=['z'],right_on=['z'])

                print(df_merge)
                cols = []
                for b in self.bands:
                    df_merge['flux_5_e_sec_{}'.format(b)]=df_merge['flux_5_e_sec_{}'.format(b)]/np.sqrt(df_merge['cadence'])
                    df_merge['m5calc_{}'.format(b)] = self.f5_to_m5[b](df_merge['flux_5_e_sec_{}'.format(b)])
                    df_merge.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(df_merge['m5calc_{}'.format(b)]-df_merge['m5single_{}'.format(b)]))
                    
                    
                
                """
                fig, ax = plt.subplots()
                figb, axb = plt.subplots()
                for b in 'grizy':
                    test['flux_5_e_sec_{}'.format(b)]=test['flux_5_e_sec_{}'.format(b)]/np.sqrt(cadence)
                    test['m5calc_{}'.format(b)] = self.f5_to_m5[b](test['flux_5_e_sec_{}'.format(b)])
                    test.loc[:,'Nvisits_{}'.format(b)]=10**(0.8*(test['m5calc_{}'.format(b)]-test['m5single_{}'.format(b)]))
                    ax.plot(test['z'],test['Nvisits_{}'.format(b)],color=filtercolors[b])
                    axb.plot(test['z'],test['m5calc_{}'.format(b)],color=filtercolors[b])
                ax.grid()
                axb.grid()
                """
                #print(test)

                #plt.show()
    

        self.map_cad_m5 = df_merge
    
    def plot_map(self, dft):

        
        mag_range = [23., 27.5] 
        dt_range=[0.5, 20.]

        dt = np.linspace(dt_range[0], dt_range[1], 100)
        m5 = np.linspace(mag_range[0], mag_range[1], 100)
        
        
        zrange = np.arange(0.3,0.9,0.1)

        df = pd.DataFrame()
        for z in zrange:
            idb = np.abs(dft['z']-z)<1.e-5
            df = pd.concat([df,dft[idb]], sort=False)


        for b in self.bands: 

            f5 = self.m5_to_f5[b](m5)
            M5, DT = np.meshgrid(m5, dt)            
            F5, DT = np.meshgrid(f5, dt)
            metric = np.sqrt(DT) * F5

            fig = plt.figure(figsize=(8, 6))
            fig.suptitle('{} band'.format(b))
            plt.imshow(metric, extent=(mag_range[0],mag_range[1],dt_range[0],dt_range[1]), 
                       aspect='auto', alpha=0.25)

            idx = np.abs(df['cadence']-1.)<1.e-5
            dfsel = df[idx]

            dfsel = dfsel.sort_values(by=['flux_5_e_sec_{}'.format(b)])
            
            ll = dfsel['flux_5_e_sec_{}'.format(b)].values
           
            cs = plt.contour(M5, DT, metric, ll,colors='k')
            
            fmt = {}
            strs = ['$z=%3.2f$' % zz for zz in dfsel['z'].values]
            for l, s in zip(cs.levels, strs):
                fmt[l] = s
            plt.clabel(cs, inline=True, fmt=fmt,
                       fontsize=16, use_clabeltext=True)

           
            
            #for z in df['z'].unique():
            #    sel = df[np.abs(df['z']-z)<1.e-5]
            #    plt.plot(sel['m5calc_{}'.format(b)],sel['cadence'],'ro',mfc='None')
            
            plt.xlabel('$m_{5\sigma}$', fontsize=18)
            plt.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
            plt.xlim(mag_range)
            plt.ylim(dt_range)
            plt.grid(1)
            
        #plt.show()
    

    def plot(self, data, cadences, what='m5_calc',legy='m5'):

        fig, ax = plt.subplots()
        axb = None
        ls = ['-','--','-.']

        if what == 'Nvisits':
            figb, axb = plt.subplots()
            #cols = []
            #for b in self.bands:
            #    cols.append('Nvisits_{}'.format(b))

        for io,cad in enumerate(cadences):
            idx = np.abs(data['cadence']-cad)<1.e-5
            sel = data[idx]

            for b in self.bands:
                if io == 0:
                    ax.plot(sel['z'].values,sel['{}_{}'.format(what,b)].values,color=filtercolors[b],label=b,ls=ls[io])
                else:
                   ax.plot(sel['z'].values,sel['{}_{}'.format(what,b)].values,color=filtercolors[b],ls=ls[io]) 

            if what == 'Nvisits':
                #estimate the total number of visits
                #sel.loc[:,'Nvisits'] = sel[cols].sum(axis=1)
                axb.plot(sel['z'].values,sel['Nvisits'].values,
                         color='k',ls=ls[io],
                         label='cadence: {} days'.format(int(cad)))

        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))
        
        if axb:
            axb.legend()
            axb.grid()
            axb.set_xlabel(r'z')
            axb.set_ylabel(r'{}'.format(legy))



    def make_combi(self,zv, cad):

        df = pd.DataFrame()

        for z in zv:
            dfi = cad.copy()
            dfi.loc[:,'z'] = z
            df = pd.concat([df,dfi],ignore_index=True)

        return df

    def transform(self, med_m5):
 
        dictout = {}

        for band in 'grizy':
            idx = med_m5['filter'] == band
            sel = med_m5[idx]
            if len(sel)>0:
                dictout[band] = [np.median(sel['fiveSigmaDepth'].values)]
            else:
                dictout[band] = [0.0]

        
        
        return pd.DataFrame({'m5single_g': dictout['g'],
                             'm5single_r': dictout['r'],
                             'm5single_i': dictout['i'],
                             'm5single_z': dictout['z'],
                             'm5single_y': dictout['y']})
        """
        return pd.DataFrame({'m5single_g': [24.331887],
                             'm5single_r': [23.829639],
                             'm5single_i': [23.378532],
                             'm5single_z': [22.861743],
                             'm5single_y': [22.094841]})
        """
    """
    def transform_old(self, med_m5):

        df = med_m5.copy()
        
        #gr = df.groupby(['fieldname','season']).apply(lambda x: self.horizont(x))

        gr = df.groupby(['fieldname']).apply(lambda x: self.horizont(x)).reset_index()

        
        if 'season' not in gr.columns:
            gr.loc[:,'season'] = 0

        
        return pd.DataFrame({'fieldname': ['all'],
                             'season': [0],
                             'm5single_g': [0.],
                             'm5single_r': [23.9632],
                             'm5single_i': [23.5505],
                             'm5single_z': [23.0003],
                             'm5single_y': [22.2338]})
        
        return pd.DataFrame({'fieldname': ['all'],
                             'season': [0],
                             'm5single_g': [0.],
                             'm5single_r': [23.829639],
                             'm5single_i': [23.378532],
                             'm5single_z': [22.861743],
                             'm5single_y': [22.094841]})
        """
    def horizont(self,grp):
        
        dictout = {}

        for band in 'grizy':
            idx = grp['filter'] == band
            sel = grp[idx]
            if len(sel)>0:
                dictout[band] = [np.median(sel['fiveSigmaDepth'].values)]
            else:
                dictout[band] = [0.0]

        
        return pd.DataFrame({'m5single_g': dictout['g'],
                             'm5single_r': dictout['r'],
                             'm5single_i': dictout['i'],
                             'm5single_z': dictout['z'],
                             'm5single_y': dictout['y']})
        




"""
        def nvisits_deltam5(m5,m5_median):

            diff = m5-m5_median
        
            nv = 10**(0.8*diff)

            #return nv.astype(int)
            return nv
"""
