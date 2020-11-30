import os
import pandas as pd
import numpy as np

from .signal_bands import RestFrameBands, SignalBand
from .ana_file import AnaMedValues, Anadf
from .utils import flux5_to_m5, m5_to_flux5
from sn_tools.sn_io import loopStack
from . import plt, filtercolors


class Data:
    def __init__(self, theDir, fname,
                 m5file,
                 x1=-2.0,
                 color=0.2,
                 blue_cutoff=380.,
                 red_cutoff=800.,
                 error_model=0,
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
        error_model: bool, opt
          simulation with (1) or without (0) error model
        bands: str, opt
         filters to consider (default: grizy)

        """

        self.x1 = x1
        self.color = color
        self.bands = bands
        self.blue_cutoff = blue_cutoff
        self.red_cutoff = red_cutoff
        self.lcName = fname

        # load lc
        lc = self.load_data(theDir, fname)

        # estimate zcutoff by bands
        self.zband = RestFrameBands(blue_cutoff=blue_cutoff,
                                    red_cutoff=red_cutoff).zband

        # apply these cutoffs if error_model=0
        if error_model:
            self.lc = lc
        else:
            self.lc = self.wave_cutoff(lc)
            
        # get the flux fraction per band
        self.fracSignalBand = SignalBand(self.lc)

        # load median m5
        #m5_fpath = '{}/{}'.format(theDir, m5file)
        self.m5_FieldBandSeason, self.m5_FieldBand, self.m5_Band = self.load_m5(
            m5file)

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

        lcData = self.load(theDir, fname)

        # select data for the type of SN (x1,color)

        idx = (lcData['x1']-self.x1) < 1.e-1
        idx &= (lcData['color']-self.color) < 1.e-1
        lcData = lcData[idx]

        # remove lc points with negative flux

        idx = lcData['flux_e_sec'] >= 0
        lcData = lcData[idx]

        # transform initial lcData to a pandas df

        lcdf = pd.DataFrame(np.copy(lcData))
        lcdf['band'] = lcdf['band'].map(lambda x: x.decode()[-1])

        return lcdf

    def load(self, theDir, fname):
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

    def wave_cutoff(self, df):
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

        res = df.merge(zbanddf, left_on=['band', 'selband'], right_on=[
                       'band', 'selband'])

        # select obs with z > z_blue (blue cutoff)

        idx = res['z'] <= res['z_blue']
        idx &= res['z'] >= res['z_red']

        res = res[idx]

        # remove selband and z_blue and z_red
        res = res.drop(columns=['selband', 'z_blue', 'z_red'])

        return res

    def load_m5(self, m5_file):
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
        median_m5_field_filter = self.anamed.median_m5_field_filter

        median_m5_filter = median_m5_field_filter.groupby(
            ['filter'])['fiveSigmaDepth'].median().reset_index()
        #medm5_season = median_m5_season.groupby(['filter'])['fiveSigmaDepth'].median().reset_index()

        return median_m5_field_filter_season, median_m5_field_filter, median_m5_filter

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

        self.fracSignalBand.plotSignalBand(self.x1, self.color)

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
    def __init__(self, snr_calc, cadence, theDir, m5_file, m5_type, choice_type, bands):
        """
        class to estimate the number of visits
        for a given cadence

        Parameters
        ----------
        snr_calc: pandas df
         with the following columns:
          - x1: SN stretch  
          - color: SN color 
          - z: SN redshift 
          - SNRcalc_r,i,z,y: SNR for r,i,z,y bands
          - m5calc_r,i,z,y: m5 estimated from SNR for r,i,z,y bands 
          - fracSNR_r,i,z,y: SNR distribution for r,i,z,y bands 
          - flux_5_e_sec_r,i,z,y: 5-sigma flux (pe/sec) for r,i,z,y bands 
        cadence: float
         cadence value
        theDir: str
         location directory of m5 files
        m5_file: str
         m5 file name
        m5_type: str
         type of m5 values used
         eg: median_m5_field_filter_season: median m5 per field and per filter and per season
             median_m5_field_filter: median m5 per field and per filter (independent on season)
             median_m5_filter : median m5 per filter (independent on field and season)
        choice_type: str
         choice for SNR:
           - Nvisits: min tot number of visits
           - Nvisits_y: min tot number of visits in y
           - fracflux : SNR distrib per band close to flux fraction dist per band
        bands: str
         bands to consider
        """

        outName = 'Nvisits_cadence_{}_{}.npy'.format(choice_type, m5_type)

        if not os.path.isfile(outName):
            # file not found
            # calculation necessary

            # grab values
            self.cadence = cadence  # cadence
            self.snr_calc = snr_calc  # snrs

            # new columns for output df
            self.cols = ['z', 'Nvisits', 'cadence']
            for band in bands:
                self.cols.append('Nvisits_{}'.format(band))

            # get m5 values
            m5_fpath = '{}/{}'.format(theDir, m5_file)
            medclass = AnaMedValues(m5_fpath)
            m5 = eval('{}.{}'.format('medclass', m5_type))

            # df_tot: output df
            df_tot = pd.DataFrame()

            # groupby fieldname and season - apply getVisits for the groups
            df_tot = m5.groupby(['fieldname', 'season']).apply(
                lambda x: self.getVisits(x)).reset_index()

            self.nvisits_cadence = df_tot

            # save as numpy record
            np.save(outName, np.copy(df_tot.to_records(index=False)))

        else:
            # file found: just load it!
            self.nvisits_cadence = pd.DataFrame(
                np.load(outName, allow_pickle=True))

    def getVisits(self, grp):
        """
        Method to get the number of visits for the group grp
        Calls the class Nvisits_m5

        Parameters
        ----------
        grp: group (in the pandas df group sense)

        Returns
        -------
        df: pandas df with the following columns:
         - x1: SN stretch 
         - color: SN color
         - z: SN redshift 
         - SNRcalc_g,r,i,z,y: SNR for g,r,i,z,y bands
         - m5calc_g,r,i,z,y: m5 from SNR for g,r,i,z,y bands
         - fracSNR_g,r,i,z,y: SNR distribution for g,r,i,z,y bands
         - flux_5_e_sec_g,r,i,z,y: 5-sigma flux (pe/sec) for g,r,i,z,y bands
         - idx:
         - m5single_g,r,i,z,y: m5 single exposure for g,r,i,z,y bands
         - cadence: cadence
         - Nvisits_g,r,i,z,y: total number of visits for g,r,i,z,y bands
        """

        # get the number of visits
        df = Nvisits_m5(self.snr_calc, grp).nvisits

        # select data corresponding to the cadence
        if self.cadence > 0.:
            io = np.abs(df['cadence']-self.cadence) < 1.e-5
            # print('iiii',df.columns,self.cols)

            df = df.loc[io, self.cols]
        else:
            df = df.loc[:, self.cols]

        return df

    def plot(self):
        """
        Method to plot the Delta number of visits
        as a function of the redshift

        These plots (one per field) illustrate how
        the number of visits may vary
        depending on the season (that is on the reference m5 values)

        """

        # this for the plot
        print(self.nvisits_cadence.groupby(['fieldname', 'z']).apply(
            lambda x: np.min(x['Nvisits'])).reset_index())

        df = self.nvisits_cadence.groupby(['fieldname', 'z']).agg(
            {'Nvisits': ['min', 'max', 'median']})
        # rename columns
        df.columns = ['Nvisits_min', 'Nvisits_max', 'Nvisits_median']

        # reset index to get grouped columns back
        df = df.reset_index()

        for fieldname in df['fieldname'].unique():
            io = df['fieldname'] == fieldname
            sel = df[io]
            fig, ax = plt.subplots()
            fig.suptitle(fieldname)

            ax.fill_between(sel['z'], sel['Nvisits_min'],
                            sel['Nvisits_max'], color='grey')
            ax.plot(sel['z'], sel['Nvisits_median'], color='k')

            ax.grid()
            ax.set_xlim([0.3, 0.85])
            ax.set_xlabel(r'z')
            ax.set_ylabel(r'Number of visits')

            figb, axb = plt.subplots()
            figb.suptitle(fieldname)

            axb.fill_between(sel['z'], sel['Nvisits_min']-sel['Nvisits_median'],
                             sel['Nvisits_max']-sel['Nvisits_median'], color='grey')
            # axb.plot(sel['z'],sel['Nvisits_median'],color='k')

            axb.grid()
            axb.set_xlim([0.3, 0.85])
            axb.set_xlabel(r'z')
            axb.set_ylabel(r'$\Delta$Number of visits')


class Nvisits_m5:
    def __init__(self, tab, med_m5):
        """
        class to estimate the number of visits 
        requested to "reach" m5 values 
        depending on cadence

        Parameters
        ----------

        tab: panda df with the following columns:
         - x1: SN stretch 
         - color: SN color
         - z: SN redshift 
         - SNRcalc_g,r,i,z,y: SNR for g,r,i,z,y bands
         - m5calc_g,r,i,z,y: m5 from SNR for g,r,i,z,y bands
         - fracSNR_g,r,i,z,y: SNR distribution for g,r,i,z,y bands
         - flux_5_e_sec_g,r,i,z,y: 5-sigma flux (pe/sec) for g,r,i,z,y bands
        med_m5: panda df with m5 values (colname: fiveSigmaDepth) 

        """

        cols = tab.columns[tab.columns.str.startswith('m5calc')].values

        self.bands = ''.join([col.split('_')[-1] for col in cols])
        # get flux5 to m5 and m5 to flux5 conversions
        self.f5_to_m5 = flux5_to_m5(self.bands)
        self.m5_to_f5 = m5_to_flux5(self.bands)
        # snr reference values to estimate m5
        self.snr = tab

        # transform m5 info: for columns to rows
        self.med_m5 = self.transform(med_m5)

        # select data with z>=0.2
        idx = tab['z'] >= 0.2
        tab = tab[idx]

        # get the number of visits
        self.nvisits = self.estimateVisits()

        # This is to plot z isocurve in the plane(cad obs frame,m5)
        """
        self.plot_map(self.nvisits) 
        cads = list(np.arange(0.,5.,1.))
        self.plot(self.nvisits,cads)
        plt.show()
        """

    def estimateVisits(self):
        """
        Method to estimate the number of visits
        necessary to "reach" m5 values
        defined by SNRs(per band)
        These numbers are estimated for a set of cadences
        """

        dict_cad_m5 = {}

        # define the cadences
        cads = pd.DataFrame(np.arange(0., 10., 1.), columns=['cadence'])

        idx = 0

        # add and index to both df for the merging

        m5 = self.med_m5.copy()
        snr = self.snr.copy()

        m5.loc[:, 'idx'] = idx
        snr.loc[:, 'idx'] = idx

        # perform the merging between SNR values and m5 reference (single exp) values
        snr = snr.merge(m5, left_on=['idx'], right_on=['idx'])

        zvals = snr['z'].unique()

        # make all possible combinations of (z,cadence) pairs
        df_combi = self.make_combi(zvals, cads)

        # merge this with snr
        df_merge = snr.merge(df_combi, left_on=['z'], right_on=['z'])

        cols = []
        # estimate the number of visits per band
        for b in self.bands:
            df_merge['flux_5_e_sec_{}'.format(b)] = df_merge['flux_5_e_sec_{}'.format(
                b)]/np.sqrt(df_merge['cadence'])
            df_merge['m5calc_{}'.format(b)] = self.f5_to_m5[b](
                df_merge['flux_5_e_sec_{}'.format(b)])
            df_merge.loc[:, 'Nvisits_{}'.format(
                b)] = 10**(0.8*(df_merge['m5calc_{}'.format(b)]-df_merge['m5single_{}'.format(b)]))
            cols.append('Nvisits_{}'.format(b))
            df_merge['Nvisits_{}'.format(
                b)] = df_merge['Nvisits_{}'.format(b)].fillna(0.0)

        # estimate the total number of visits
        df_merge.loc[:, 'Nvisits'] = df_merge[cols].sum(axis=1)

        return df_merge

    def plot_map(self, dft):
        """
        Method to plot z-curves(per band) corresponding to sigma_C<0.04
        in the plane (cadence obs frame, m5)

        Parameters
        ----------
        dft: pandas df with the following columns:
         - x1: SN stretch 
         - color: SN color
         - z: SN redshift 
         - SNRcalc_g,r,i,z,y: SNR for g,r,i,z,y bands
         - m5calc_g,r,i,z,y: m5 from SNR for g,r,i,z,y bands
         - fracSNR_g,r,i,z,y: SNR distribution for g,r,i,z,y bands
         - flux_5_e_sec_g,r,i,z,y: 5-sigma flux (pe/sec) for g,r,i,z,y bands
         - idx:
         - m5single_g,r,i,z,y: m5 single exposure for g,r,i,z,y bands
         - cadence: cadence
         - Nvisits_g,r,i,z,y: total number of visits for g,r,i,z,y bands

        """

        mag_range = [23., 27.5]
        dt_range = [0.5, 20.]

        dt = np.linspace(dt_range[0], dt_range[1], 100)
        m5 = np.linspace(mag_range[0], mag_range[1], 100)

        zrange = np.arange(0.3, 0.9, 0.1)

        df = pd.DataFrame()
        for z in zrange:
            idb = np.abs(dft['z']-z) < 1.e-5
            df = pd.concat([df, dft[idb]], sort=False)

        for b in self.bands:

            f5 = self.m5_to_f5[b](m5)
            M5, DT = np.meshgrid(m5, dt)
            F5, DT = np.meshgrid(f5, dt)
            metric = np.sqrt(DT) * F5

            fig = plt.figure(figsize=(8, 6))
            fig.suptitle('{} band'.format(b))
            plt.imshow(metric, extent=(mag_range[0], mag_range[1], dt_range[0], dt_range[1]),
                       aspect='auto', alpha=0.25)

            idx = np.abs(df['cadence']-1.) < 1.e-5
            dfsel = df[idx]

            dfsel = dfsel.sort_values(by=['flux_5_e_sec_{}'.format(b)])

            ll = dfsel['flux_5_e_sec_{}'.format(b)].values

            cs = plt.contour(M5, DT, metric, ll, colors='k')

            fmt = {}
            strs = ['$z=%3.2f$' % zz for zz in dfsel['z'].values]
            for l, s in zip(cs.levels, strs):
                fmt[l] = s
            plt.clabel(cs, inline=True, fmt=fmt,
                       fontsize=16, use_clabeltext=True)

            # for z in df['z'].unique():
            #    sel = df[np.abs(df['z']-z)<1.e-5]
            #    plt.plot(sel['m5calc_{}'.format(b)],sel['cadence'],'ro',mfc='None')

            plt.xlabel('$m_{5\sigma}$', fontsize=18)
            plt.ylabel(r'Observer frame cadence $^{-1}$ [days]', fontsize=18)
            plt.xlim(mag_range)
            plt.ylim(dt_range)
            plt.grid(1)

        # plt.show()

    def plot(self, data, cadences, what='m5calc', legy='m5'):
        """
        Method to plot m5 (per band, corresponding to sigma_C<0.04)
        as a function of the redshift
        x-axis is the redshift labelled as z.

        Parameters
        ----------
        data: pandas df with the following columns:
         - x1: SN stretch 
         - color: SN color
         - z: SN redshift 
         - SNRcalc_g,r,i,z,y: SNR for g,r,i,z,y bands
         - m5calc_g,r,i,z,y: m5 from SNR for g,r,i,z,y bands
         - fracSNR_g,r,i,z,y: SNR distribution for g,r,i,z,y bands
         - flux_5_e_sec_g,r,i,z,y: 5-sigma flux (pe/sec) for g,r,i,z,y bands
        cadences: list of float
         list of cadence values for the plot
        what: str, opt
         parameter to plot (default = m5calc)
        legy: str,opt
         y-axis legend (default: m5)



        """

        fig, ax = plt.subplots()
        axb = None
        ls = ['-', '--', '-.', '-', '-']

        if what == 'Nvisits':
            figb, axb = plt.subplots()
            #cols = []
            # for b in self.bands:
            #    cols.append('Nvisits_{}'.format(b))

        for io, cad in enumerate(cadences):
            idx = np.abs(data['cadence']-cad) < 1.e-5
            sel = data[idx]

            for b in self.bands:
                toplot = '{}_{}'.format(what, b)
                if io == 0:
                    ax.plot(sel['z'].values, sel[toplot].values,
                            color=filtercolors[b], label=b, ls=ls[io])
                else:
                    ax.plot(sel['z'].values, sel[toplot].values,
                            color=filtercolors[b], ls=ls[io])

            if what == 'Nvisits':
                # estimate the total number of visits
                #sel.loc[:,'Nvisits'] = sel[cols].sum(axis=1)
                axb.plot(sel['z'].values, sel['Nvisits'].values,
                         color='k', ls=ls[io],
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

    def make_combi(self, zv, cad):
        """
        Method to combine redshift and cadence values

        Parameters
        ----------
        zv: redshift values
        cad: cadence values

        Returns
        -------
        pandas df with all possible combinations of (redshift, cadence) pairs

        """

        df = pd.DataFrame()

        for z in zv:
            dfi = cad.copy()
            dfi.loc[:, 'z'] = z
            df = pd.concat([df, dfi], ignore_index=True)

        return df

    def transform(self, med_m5):
        """
        Method to transform m5 data to a single row


        Parameters
        ----------
        med_m5: panda df with m5 values (colname: fiveSigmaDepth)

        Returns
        -------
        pandas df with:
         m5single_g,r,i,z,y: m5 single-exposure for g,r,i,z,y bands
        """

        dictout = {}

        for band in 'grizy':
            idx = med_m5['filter'] == band
            sel = med_m5[idx]
            if len(sel) > 0:
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


class Mod_z:
    """
    Method to modify values of input file

    """

    def __init__(self, fName):

        # load the file
        tab = np.load(fName, allow_pickle=True)

        tabdf = pd.DataFrame(np.copy(tab))

        tabmod = tabdf.groupby(['cadence']).apply(
            lambda x: self.mod(x)).reset_index()
        ll = []
        for b in 'grizy':
            ll.append('Nvisits_{}'.format(b))
        tabmod['Nvisits'] = tabmod[ll].sum(axis=1)

        self.nvisits = tabmod

    def mod(self, grp):
        """
        Method to modify a group


        Parameters
        --------------
        grp : pandas df group

        Returns
        -----------
        modified grp group
        """
        for band in 'grizy':
            what = 'Nvisits_{}'.format(band)
            idx = grp[what] > 1.e-21
            idx &= grp[what] < 1.
            grp.loc[idx, what] = 1.

            if band == 'g' or band == 'r':
                Index_label = grp[grp[what] < 1.e-10].index.tolist()
                Index_label_p = grp[grp[what] > 1.e-10].index.tolist()
                #print('index', Index_label, Index_label_p)
                grp.loc[Index_label, what] = grp.loc[Index_label_p[-1]][what]
            #print('io', grp[what])

        return grp
