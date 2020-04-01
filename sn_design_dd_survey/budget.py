import yaml
from scipy import interpolate
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from . import filtercolors
from . import plt


class DD_Budget:
    def __init__(self, configName, df_visits_ref, df_visits, runtype='Nvisits_single'):
        """
        class estimating the DD budget vs redshift limit

        Parameters
        ---------------
        configName: str
         configuration (yaml) file of the DDF
        df_visits_ref: pandas df
          with the following columns
          fieldname: name of the DD field
          season: season number
          Nvisits: total number of visits
          cadence: candence
          Nvisits_g,r,i,z,y: number of visits in g,r,i,z,y bands
         df_visits: pandas df
          with same infos as df_visits_ref
         runtype: str, opt
           type of SNR run to consider (default: Nvisits_single)

        """

        # loading the configuration file for this scenario
        config = yaml.load(open(configName), Loader=yaml.FullLoader)
        self.conf = config
        # loading the number of visits for the case a single  m5 per band
        self.df_visits = df_visits
        self.nvisits, self.z, self.nvisits_band = self.interp_visits(
            df_visits, runtype='')

        # loading the number of visits for the case one m5 per band per season and per field
        self.df_visits_ref = df_visits_ref
        self.nvisits_ref, self.z_ref, self.nvisits_band_ref = self.interp_visits(
            df_visits_ref, runtype='Nvisits_single')

        # estimate the budget

        self.budget = self.budget_calc(runtype)
        self.runtype = runtype
        # if self.runtype == 'Nvisits_single':
        self.summary_Nvisits_single()

    def interp_visits(self, df_tot, runtype):
        """
        Method to interpolate the number of visits vs z

        Parameters
        ---------
        df_tot: pandas df
         data used to make interpolations
        run_type: str
         type of run: Nvisits_single or Nvisits_adjusted

        Returns
        ------
        nvisits: dict of interp1d
          keys: fieldName, season; parameter: z
        z: dict of interp1d
          keys: fieldName, season; parameter: nvisits
        nvisits_band: dict of interp1d
          keys: fieldName, season, band; parameter: z 

        """

        fieldnames = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAIS', 'ADFS1', 'ADFS2']
        seasons = range(1, 11)
        nvisits = {}
        nvisits_band = {}
        z = {}
        for fieldname in fieldnames:
            nvisits[fieldname] = {}
            nvisits_band[fieldname] = {}
            z[fieldname] = {}
            for season in seasons:
                fieldname_ref = fieldname
                season_ref = season
                if runtype == 'Nvisits_single':
                    fieldname_ref = 'all'
                    season_ref = 0
                idx = df_tot['fieldname'] == fieldname_ref
                idx &= df_tot['season'] == season_ref
                idx &= np.abs(df_tot['cadence'] -
                              self.conf[fieldname]['cadence']) < 1.e-5
                sel = df_tot[idx]

                nvisits[fieldname][season] = interpolate.interp1d(
                    sel['z'], sel['Nvisits'], bounds_error=False, fill_value=0.0)
                z[fieldname][season] = interpolate.interp1d(
                    sel['Nvisits'], sel['z'], bounds_error=False, fill_value=0.0)
                nvisits_band[fieldname][season] = {}

                for b in 'grizy':
                    nvisits_band[fieldname][season][b] = interpolate.interp1d(
                        sel['z'], sel['Nvisits_{}'.format(b)], bounds_error=False, fill_value=0.0)

        return nvisits, z, nvisits_band

    def interp_ref(self, df_ref, what='Nvisits'):

        idx = df_ref['fieldname'] == 'all'
        idx &= df_ref['season'] == 0
        sel = df_ref[idx]

        nvisits_ref = interpolate.interp1d(
            sel['z'], sel[what], bounds_error=False, fill_value=0.0)

        return nvisits_ref

    def budget_calc(self, runtype):
        """
        Method to estimate, vs z, the DD budget

        Parameters
        ----------
        run_type: str
         type of run: Nvisits_single or Nvisits_adjusted

        Returns
        -------
        pandas df with the following cols:
         Nvisits_fieldName_season: number of visits 
                                   for all field/season considered in the scenario (conf file)
         z_fieldName_season: zlimit for all field/season considered in the scenario (conf file)
         z_ref: redshift limit corresponding to the case same number of visits per field/season/night
         Nvisits: total number of visits
         Nvisits_night: total number of visits per night
         DD_budget: DD budget

        """

        zr = np.arange(0.1, 0.9, 0.05)
        df_tot = pd.DataFrame()

        """
        if runtype == 'Nvisits_single':
            Nvisits_ref = self.nvisits_ref(zr)
        """
        self.cols = []
        self.cols_night = []
        self.zcols = []
        for fieldname in self.conf['Fields']:
            theconf = self.conf[fieldname]
            for season in theconf['seasons']:
                if runtype == 'Nvisits_single':
                    nvisits_night = self.nvisits_ref[fieldname][season](zr)
                else:
                    nvisits_night = self.nvisits[fieldname][season](zr)
                nvisits_season = nvisits_night*30 * \
                    self.conf[fieldname]['season_length'] / \
                    self.conf[fieldname]['cadence']

                zvals = self.z[fieldname][season](nvisits_night)
                zname = '{}_{}_{}'.format('z', fieldname, season)
                vname = '{}_{}_{}'.format('Nvisits', fieldname, season)
                vname_night = '{}_{}_{}'.format(
                    'Nvisits_night', fieldname, season)
                self.cols.append(vname)
                self.cols_night.append(vname_night)
                self.zcols.append(zname)
                dfa = pd.DataFrame(nvisits_season, columns=[vname])
                dfa.loc[:, zname] = zvals
                dfa.loc[:, vname_night] = nvisits_night
                dfa.loc[:, 'zref'] = zr
                if df_tot.empty:
                    df_tot = dfa.copy()
                else:
                    df_tot = df_tot.merge(
                        dfa, left_on=['zref'], right_on=['zref'])

        df_tot['Nvisits'] = df_tot[self.cols].sum(axis=1)
        df_tot['Nvisits_night'] = df_tot[self.cols_night].median(axis=1)
        df_tot['DD_budget'] = df_tot['Nvisits']/self.conf['Nvisits']

        return df_tot

    def summary_Nvisits_single(self):
        """
        Method to estimate a summary of the budget results regarding
        the number of visits and the redshift limits.

        """

        z = np.arange(0.1, 0.90, 0.01)
        idx = (self.budget['DD_budget'] < 0.15)
        idx &= (self.budget['DD_budget'] >= 0.)
        toplot = self.budget[idx]

        medz = toplot[self.zcols].median(axis=1)  # median redshift limit
        medbud = toplot['DD_budget']
        medvisits = toplot['Nvisits_night']

        # get zlim_min and zlim_mac cols
        r = []

        # buds = np.arange(0.000001,0.10,0.0001)

        df_bud_z = pd.DataFrame()

        for io, col in enumerate(self.zcols):
            idx = toplot[col] >= 0.1
            idx &= toplot[col] <= 0.85
            # plt.plot(toplot[idx][col],toplot[idx]['DD_budget'],color='k')
            """
            interp_ddbudget = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                        toplot[idx][col],
                                                        bounds_error=False,fill_value=0)
            interp_night = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                        toplot[idx]['Nvisits_night'],
                                                        bounds_error=False,fill_value=0)
            """
            interp_ddbudget = interpolate.interp1d(toplot[idx][col],
                                                   toplot[idx]['DD_budget'],
                                                   bounds_error=False, fill_value=0)

            interp_night = interpolate.interp1d(toplot[idx]['DD_budget'],
                                                toplot[idx]['Nvisits_night'],
                                                bounds_error=False, fill_value=0)

            df = pd.DataFrame()
            df.loc[:, 'budget'] = interp_ddbudget(z)
            df.loc[:, 'z'] = z
            df.loc[:, 'name'] = col
            df.loc[:, 'Nvisits_night'] = interp_night(df['budget'])

            df_bud_z = pd.concat([df_bud_z, df], sort=False)

            # r.append((col,interp_ddbudget(0.04),interp_ddbudget(0.08)))

        df_buz_z = df_bud_z.sort_values(by=['budget'])

        df_min = df_bud_z[df_bud_z['budget'].gt(
            0.)].groupby(['z']).min().reset_index()
        df_max = df_bud_z[df_bud_z['budget'].gt(
            0.)].groupby(['z']).max().reset_index()
        df_med = df_bud_z[df_bud_z['budget'].gt(0.)].groupby(
            ['z']).median().reset_index()

        """
        colors = dict(zip(['COSMOS','CDFS','XMM-LSS','ELAIS',
                      'ADFS1','ADFS2'],['k','b','r','g','m','orange']))
        for io,col in enumerate(self.zcols):
            idx = toplot[col]>=0.3
            idx &= toplot[col]<=0.85
            fieldname = col.split('_')[1]
            plt.plot(toplot[idx][col],toplot[idx]['DD_budget'],color=colors[fieldname])

        plt.plot(df_min['z'],df_min['budget'])
        plt.plot(df_max['z'],df_max['budget'])
        plt.show()
        """
        self.interpmin = interpolate.interp1d(
            df_min['z'], df_min['budget'], bounds_error=False, fill_value=0.10)
        self.interpmax = interpolate.interp1d(
            df_max['z'], df_max['budget'], bounds_error=False, fill_value=0.0)

        self.interpmin_ddbudget = interpolate.interp1d(
            df_max['budget'], df_max['z'], bounds_error=False, fill_value=0.10)
        self.interpmax_ddbudget = interpolate.interp1d(
            df_min['budget'], df_min['z'], bounds_error=False, fill_value=0.0)

        self.interp_ddbudget = interpolate.interp1d(
            df_med['budget'], df_med['z'], bounds_error=False, fill_value=0.0)
        self.interp_z_ddbudget = interpolate.interp1d(
            df_med['z'], df_med['budget'], bounds_error=False, fill_value=0.0)
        self.interp_z = interpolate.interp1d(
            df_med['z'], df_med['Nvisits_night'], bounds_error=False, fill_value=0.0)

        """
        self.interpmin = interpolate.interp1d(
            toplot[colmin],medbud,bounds_error=False,fill_value=0.10)
        self.interpmax = interpolate.interp1d(
            toplot[colmax],medbud,bounds_error=False,fill_value=0.0)

        self.interpmin_ddbudget = interpolate.interp1d(
            medbud,toplot[colmin],bounds_error=False,fill_value=0.10)
        self.interpmax_ddbudget = interpolate.interp1d(
            medbud,toplot[colmax],bounds_error=False,fill_value=0.0)

        self.interp_ddbudget = interpolate.interp1d(
            medbud,medz,bounds_error=False,fill_value=0.0)
        self.interp_z =  interpolate.interp1d(
            medz,medvisits,bounds_error=False,fill_value=0.0)
        """
        self.medbud = df_med['budget'].values
        self.medz = df_med['z'].values
        self.medvisits = df_med['Nvisits_night'].values
        self.zmax = df_max['z'].max()

    def zlim_Nvisits_single(self, dd_value):
        """
        Method to estimate some results corresponding to a given DD budget

        Parameters
        ----------
        dd_value: float
         DD budget

        Returns:
        -------
        zlim_median: float
          median redshift limit
        zlim_min: float
          min redshift limit
        zlim_max: float
          max redshift limit
        nvisits_choice: float
          number of visits (per night) corresponding to zlim_median
        Nvisits_band: dict
          number of visits per night and per band (key)
        """

        if self.runtype == 'Nvisits_single':
            zlim_median = self.interp_ddbudget(dd_value)
            zlim_min = self.interpmin_ddbudget(dd_value)
            zlim_max = self.interpmax_ddbudget(dd_value)
        else:
            nVisits = self.nVisits_Fields(dd_value)
            for key, vals in nVisits.items():
                for keyb, valb in vals.items():
                    zlim_median = np.median(valb['zref'])
                    zlim_min = np.min(valb['zref'])
                    zlim_max = np.max(valb['zref'])
                    #self.zmax = zlim_max

        """
        nvisits_choice = self.interp_z(zlim_median)
        Nvisits_band = {}
        for b in 'rizy':
            myinterp = self.interp_ref(
                self.df_visits_ref, 'Nvisits_{}'.format(b))
            Nvisits_band[b] = myinterp(zlim_median)

        return zlim_median, zlim_min, zlim_max, nvisits_choice, Nvisits_band
        """
        return zlim_median, zlim_min, zlim_max

    def nVisits_Fields(self, dd_value):
        """
        Method to estimate the number of visits per fields
        depending on the dd value

        Parameters
        ---------------
        dd_value: float
          DD budget

        Returns
        ----------
        nVisits: dict
          dict with the number of visits (per obs night)

        """
        nVisits = {}
        # print(self.budget.columns)
        for fieldName in self.conf['Fields']:
            nVisits[fieldName] = {}
            theconf = self.conf[fieldName]
            for seas in theconf['seasons']:
                nVisits[fieldName][seas] = {}
                zName = 'z_{}_{}'.format(fieldName, seas)
                nvisitsName = 'Nvisits_{}_{}'.format(fieldName, seas)
                myinterp = interpolate.interp1d(
                    self.budget['DD_budget'].values, self.budget['Nvisits_night_{}_{}'.format(fieldName, seas)].values)
                nVisits[fieldName][seas]['all'] = np.asscalar(
                    myinterp(dd_value))
                myinterpz = interpolate.interp1d(
                    self.budget['DD_budget'].values, self.budget[zName].values)
                nVisits[fieldName][seas]['zlim'] = np.asscalar(
                    myinterpz(dd_value))

                myinterpz = interpolate.interp1d(
                    self.budget['DD_budget'].values, self.budget['zref'].values)
                nVisits[fieldName][seas]['zref'] = np.asscalar(
                    myinterpz(dd_value))

                for b in 'grizy':
                    if self.runtype == 'Nvisits_single':
                        nVisits[fieldName][seas][b] = np.asscalar(self.nvisits_band_ref[fieldName][seas][b](
                            myinterpz(dd_value)))
                    else:
                        nVisits[fieldName][seas][b] = np.asscalar(self.nvisits_band[fieldName][seas][b](
                            myinterpz(dd_value)))

        return nVisits

    def plot_budget_zlim(self, dd_budget=-1):
        """
        Plot to display DD budget results as a function of the redshift limit
        if dd_budget>0: an estimation of zlim (min, median, max) corresponding to dd_budget 
        is displayed 

        Parameters
        ----------
        dd_budget: float, opt
         DD budget (default: -1)

        """

        zminval = 0.1
        z = np.arange(zminval, 0.9, 0.05)
        if dd_budget > 0.:
            zlim_median, zlim_min, zlim_max = self.zlim_Nvisits_single(
                dd_budget)

        # Now plot

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig1.suptitle(self.conf['confName'])
        #ax2.set_title('{} - season {}'.format(fieldName,season))
        ax1.set_ylim(ymax=0.10)

        ax1.set_xlim([zminval+0.01, self.zmax])
        ax1.set_ylim([self.interp_z_ddbudget(zminval), np.min(
            [0.10, self.interp_z_ddbudget(self.zmax)])])
        zb = np.arange(zminval, self.zmax, 0.01)
        ax1.fill_between(zb, self.interpmin(
            zb), self.interpmax(zb), color='yellow', alpha=0.5)
        ax1.plot(self.medz, self.medbud, color='k')
        ax1.set_ylabel(r'DD budget')
        ax1.set_xlabel(r'z$_{lim}$')
        ax1.grid()

        if dd_budget > 0.:
            # draw arrows corresponding to zlim (min, max, median) values
            ax1.plot(ax1.get_xlim(), [dd_budget]*2, color='r', ls='--')
            for ip, val in enumerate([('min', zlim_min), ('max', zlim_max), ('median', zlim_median)]):
                ax1.arrow(val[1], dd_budget, 0., -dd_budget,
                          length_includes_head=True, color='b',
                          head_length=0.005, head_width=0.01)
                ax1.text(0.35, 0.04-0.005*ip,
                         '$z_{lim}^{'+val[0]+'}$='+str(np.round(val[1], 2)))

    def plot_budget_visits(self, fieldName, season, dd_budget=-1):
        """
        Plot to display DD budget results 
        The plot has two parts:
        - left side: DD budget vs zlim for the field considered
        - right side: Number of visits vs redshift limit for the field considered

        Parameters
        ----------
        dd_budget: float
        DD budget
        fieldName: str
        name of the field to display
        season:
        name of the season to display 
        dd_budget: float, opt
        DD budget (default: -1)
        """

        zmin = 0.1
        zmax = 0.9
        z = np.arange(zmin, zmax, 0.05)
        if dd_budget > 0.:
            # get the number of visits for the fields

            nVisits = self.nVisits_Fields(dd_budget)
            nvisits_choice = nVisits[fieldName][season]['all']
            Nvisits_band = nVisits
        # Now plot

        interp_bud_z = interpolate.interp1d(
            self.budget['DD_budget'].values,
            self.budget['z_{}_{}'.format(fieldName, season)].values,
            bounds_error=False, fill_value=0.0)

        interp_z_bud = interpolate.interp1d(self.budget['z_{}_{}'.format(fieldName, season)].values,
                                            self.budget['DD_budget'.format(
                                                fieldName, season)].values,
                                            bounds_error=False, fill_value=0.0)

        interp_z_visits_field = interpolate.interp1d(self.budget['z_{}_{}'.format(fieldName, season)].values,
                                                     self.budget['Nvisits_night_{}_{}'.format(
                                                         fieldName, season)].values,
                                                     bounds_error=False, fill_value=0.0)
        interp_z_visits = interpolate.interp1d(self.budget['zref'],
                                               self.budget['Nvisits_night_{}_{}'.format(
                                                   fieldName, season)].values,
                                               bounds_error=False, fill_value=0.0)
        interp_visits_z = interpolate.interp1d(self.budget['Nvisits_night_{}_{}'.format(fieldName, season)].values,
                                               self.budget['zref'],
                                               bounds_error=False, fill_value=0.0)

        zb = np.arange(zmin, zmax, 0.01)
        bud = interp_z_bud(zb)
        ymax = np.min([0.1, np.max(bud)])
        zmax = zb[np.argmax(bud)]

        fig, axs = plt.subplots(
            1, 2, gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(16, 8))
        (ax1, ax2) = axs

        fig.suptitle(
            '{} - {} - season {}'.format(self.conf['confName'], fieldName, season))

        # left-handed plot: DDbudget vs zlim

        ax1.set_ylim(ymax=ymax)

        ax1.set_xlim([zmin+0.01, zmax])
        # ax1.set_ylim([interp_z_bud(zmin), np.min(
        #    [0.10, ymax])])

        budvals = interp_z_bud(zb)
        ax1.plot(zb, budvals, color='k')

        ax1.set_ylabel(r'DD budget')
        ax1.set_xlabel(r'z$_{lim}$')
        ax1.grid()

        axa = ax1.twinx()

        axa.plot(zb, budvals, color='k')

        zlims = interp_bud_z(ax1.get_ylim())
        Nvisitslim = interp_z_visits_field(zlims)

        axa.set_ylim(Nvisitslim)

        zticks = interp_bud_z(ax1.get_yticks())

        Nvisits_ticks = interp_z_visits_field(zticks)

        axa.set_yticks(Nvisits_ticks)

        if dd_budget > 0.:
            ax1.plot(ax1.get_xlim(), [dd_budget]*2, color='r', ls='--')
            zlimit = interp_bud_z(dd_budget)
            print('zlimit', zlimit, dd_budget)
            ax1.arrow(zlimit, dd_budget, 0., -dd_budget,
                      length_includes_head=True, color='b',
                      head_length=0.005, head_width=0.01)
            """
            ax1.text(0.35, 0.05,
                     '$z_{lim}$='+str(np.round(zlimit, 2)), fontsize=15)
            """
            ax1.text(1.05*zlimit, 0.5*dd_budget,
                     '$z_{lim}$='+str(np.round(zlimit, 2)), fontsize=15)
        # right-hand side plot: nvisits vs zlim
        # adjust axes
        ax2.set_ylim(axa.get_ylim())
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_yticks(axa.get_yticks())
        # make the twin to have Nvisits on the right
        ax3 = ax2.twinx()
        ax3.set_ylim(axa.get_ylim())
        ax3.set_xlim(ax1.get_xlim())
        ax3.plot(zb, interp_z_visits(zb),
                 color='k', label='all')

        for b in 'grizy':
            myinterp = self.interp_ref(
                self.df_visits_ref, 'Nvisits_{}'.format(b))
            if self.runtype == 'Nvisits_single':
                ax3.plot(z, self.nvisits_band_ref[fieldName][season][b](
                    z), color=filtercolors[b], label='{}'.format(b))
            else:
                ax3.plot(z, self.nvisits_band[fieldName][season][b](
                    z), color=filtercolors[b], label='{}'.format(b))
        ax2.grid()
        ax3.legend()
        ax2.yaxis.set_ticklabels([])
        axa.yaxis.set_ticklabels([])
        ax3.set_ylabel(r'Nvisits/night')
        ax2.set_xlabel(r'z')
        ax3.set_yticks(ax2.get_yticks())
        ax3.set_yticklabels(np.round(ax3.get_yticks()).astype(int))

        if dd_budget > 0.:

            ax2.plot(ax2.get_xlim(), [
                     nVisits[fieldName][season]['all']]*2, color='r', ls='--')
            nvisits_choice_calc = 0
            zName = 'zref'
            ax2.arrow(nVisits[fieldName][season][zName], nvisits_choice, 0.0, ax3.get_ylim()[0]-nvisits_choice,
                      length_includes_head=True, color='b',
                      head_length=1., head_width=0.01)

            for io, band in enumerate('grizy'):
                nvisits_band = int(np.round(nVisits[fieldName][season][band]))
                if nvisits_band > 0:

                    ymax = ax2.get_ylim()[1]
                    ax2.arrow(nVisits[fieldName][season][zName], nVisits[fieldName][season][band],
                              ax2.get_xlim()[
                        1]-nVisits[fieldName][season][zName], 0.,
                        length_includes_head=True, color='b',
                        head_width=1., head_length=0.01)
                    """
                    ax2.text(0.35, 0.9*nvisits_choice-0.1*io*nvisits_choice,
                             '$N_{visits}$-'+band+' = '+str(nvisits_band), fontsize=15)
                    """
                    ax2.text(0.3, 0.8*ymax-0.07*io*ymax,
                             '$N_{visits}$-'+band+' = '+str(nvisits_band), fontsize=15)

                    nvisits_choice_calc += nvisits_band

            ax2.text(0.3, 0.9*ymax,
                     '$N_{visits}$ - sum = '+str(int(nvisits_choice_calc)), fontsize=15)
            """
            ax2.text(0.35, 1.1*nvisits_choice,
                     '$N_{visits}$ - sum = '+str(int(nvisits_choice_calc)), fontsize=15)
            """

    def printVisits(self, dd_value):
        """
        Method to print(in a table) the number of visits and zlim
        per field and per night corresponding to a DD budget

        Parameters
        ----------
        dd_value: float
         DD budget

        """

        # get infos corresponding to this budget

        nVisits = self.nVisits_Fields(dd_value)

        nameConv = dict(zip(['season', 'all', 'zlim', 'zref', 'r', 'i', 'z', 'y'],
                            ['season', 'Nvisits', 'zlim', 'zref',
                             'Nvisits_r', 'Nvisits_i',
                             'Nvisits_z', 'Nvisits_y']))

        for key, vals in nVisits.items():
            fig, ax = plt.subplots()
            names = []
            vv = []
            names.append('season')
            io = -1
            for keyb, valb in vals.items():
                io += 1
                ro = [keyb]
                for keyc, valc in valb.items():
                    if keyc != 'zlim' and keyc != 'zref':
                        ro.append(int(np.round(valc)))
                    else:
                        ro.append(np.round(valc, 2))
                    if io == 0:
                        names.append(nameConv[keyc])
                vv.append(ro)

            tab = np.rec.fromrecords(vv, names=names)
            ll = '{} - {} - DD budget: {}'.format(
                self.conf['confName'], key, dd_value)
            ax.text(0.2, 0.8, ll)
            ax.table(cellText=tab, colLabels=names, loc='center')

            plt.axis('off')
