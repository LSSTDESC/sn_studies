import pandas as pd
from sn_fom import plt
import numpy as np
from scipy.interpolate import make_interp_spline
from optparse import OptionParser


def get_sigma_mb_z(fichName, dbNames=['DD_0.90', 'DD_0.85', 'DD_0.80', 'DD_0.75', 'DD_0.70', 'DD_0.65', 'DD_0.60', 'DD_0.55', 'DD_0.50']):

    vvars = ['z', 'dbName', 'sigma_mb_mean']
    prefName = 'sigma_mb_from_simu'
    for fich in fichName:
        data = pd.read_hdf('{}.hdf5'.format(fich))

        prefix = fich.split('.')[0].split('_')
        print('boo', prefix)
        prefix = '_'.join(pp for pp in prefix[-2:])
        print('allo', prefix)
        for dbName in dbNames:
            idx = data['dbName'] == dbName
            seldata = data[idx]
            df = seldata[vvars]
            df = df.rename(columns={'sigma_mb_mean': 'sigma_mb'})
            outName = '{}_{}_{}.hdf5'.format(prefName, dbName, prefix)
            df.to_hdf(outName, key='sigma_mb')


class Plot_Sigma_Components:
    """
    class to plot sigma componentsvs z for a set of redshift completeness survey

    Parameters
    ----------------
    fichname: str
      name of the file to process (plot)
    dbNames: list(str), opt
      list of redshift completeness survey to plot (default: ['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65'])
    alpha: float, opt
      alpha parameter to estimate x1 error contribution (default: 0.13)
    beta: float, opt
      alpha parameter to estimate color error contribution (default: 3.1)

    """

    def __init__(self, fichName, dbNames=['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65'], alpha=0.13, beta=3.1):

        self.alpha = alpha
        self.beta = beta
        self.dbNames = dbNames
        for fich in fichName:
            data = pd.read_hdf('{}.hdf5'.format(fich))
            self.plot_sigma_component(data)

    def plot_sigma_component(self, dd):
        """
        This is where the plot is effectively made

        Parameters
        --------------
        dd: pandas df
          data to plot

        """
        """
        zmin = 0.1
        zmax = 1.1
        nbins = 20

        dres = pd.DataFrame()
        for key, val in dd.items():
            da = binned_data(zmin, zmax, nbins, val)
            da['dbName'] = key
            dres = pd.concat((dres, da))
        """

        for dbName in self.dbNames:
            idx = dd['dbName'] == dbName
            sel = dd[idx]
            sel = sel.sort_values(by=['z'])
            sel = sel.fillna(0)
            selb = sel.to_records(index=False)
            idz = selb['z'] <= 1.09
            selb = selb[idz]
            zcomp = dbName.split('_')[-1]
            self.plot_Indiv(zcomp, selb)

    def plot_Indiv(self, zcomp, selb):

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.suptitle('$z_{complete}$='+str(zcomp))
        vvars = ['mu', 'x1', 'color', 'mb']
        coeffs = dict(zip(vvars, [1, 0.14, 3.1, 1.]))
        for vv in vvars:
            ax.plot(selb['z'], coeffs[vv]*selb['sigma_{}_mean'.format(vv)],
                    label=vv, lw=3)
        """
        ax.plot(selb['z'], selb['sigma_mu_mean'],
                label='$\mu$', lw=3)
        ax.plot(selb['z'], self.alpha*selb['sigma_x1_mean'],
                label='$\alpha x$', lw=3)
        ax.plot(selb['z'], self.beta*selb['sigma_color_mean'],
                label='$\beta c$', lw=3)
        ax.plot(selb['z'], selb['sigma_mb_mean'],
                label='$m_b$', lw=3)
        xmax = np.max(selb['z'])
        """
        """
        xnew = np.linspace(
                np.min(selb['z']), np.max(selb['z']), 100)
            selfpl = make_interp_spline(
                selb['z'], selb['sigma_mu_mean'], k=3)  # type: BSpline
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)
        """
        ax.grid()
        ax.set_ylabel('error budget')
        ax.set_xlabel('$z$')
        ax.legend()
        xmax = np.max(selb['z'])
        ax.set_xlim([0.05, xmax])
        ax.set_ylim([0.0, None])


class Plot_Sigma_mu:
    """
    class to plot sigma_mu vs z for a set of redshift completeness survey

    Parameters
    ----------------
    fichname: str
      name of the file to process (plot)
    dbNames: list(str), opt
      list of redshift completeness survey to plot (default: ['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65'])

    """

    def __init__(self, fichName, sigmamu_syste, dbNames=['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65']):

        self.dbNames = dbNames

        data = pd.read_hdf('{}.hdf5'.format(fichName[0]))

        data_syste = {}
        for fichName in sigmamu_syste:
            data_syste[fichName] = pd.read_hdf('{}.hdf5'.format(fichName))

        # estimate systematics here
        syste = self.estimate_syste(data, data_syste)
        print('syste', syste)
        self.plot_sigma_mu(data, syste)

    def plot_sigma_mu(self, dd, syste):
        """
        This is where the plot is effectively made

        Parameters
        --------------
        dd: pandas df
          data to plot

        """
        """
        zmin = 0.1
        zmax = 1.1
        nbins = 20

        dres = pd.DataFrame()
        for key, val in dd.items():
            da = binned_data(zmin, zmax, nbins, val)
            da['dbName'] = key
            dres = pd.concat((dres, da))
        """
        fig, ax = plt.subplots(figsize=(12, 9))
        ls = dict(zip(self.dbNames, ['solid', 'dotted', 'dashed', 'dashdot']))

        for dbName in self.dbNames:
            idx = dd['dbName'] == dbName
            sel = dd[idx]
            sel = sel.sort_values(by=['z'])
            sel = sel.fillna(0)
            selb = sel.to_records(index=False)
            idz = selb['z'] <= 1.09
            selb = selb[idz]
            zcomp = dbName.split('_')[-1]
            """
            ax.plot(selb['z'], selb['sigma_mu_mean'],
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)
            """
            xmax = np.max(selb['z'])

            xnew = np.linspace(
                np.min(selb['z']), np.max(selb['z']), 100)
            spl = make_interp_spline(
                selb['z'], selb['sigma_mu_mean'], k=3)  # type: BSpline
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)

            if not syste.empty:
                syste['sigma_mu_mean_plus'] = syste['sigma_mu_mean'] + \
                    syste['syste']
                syste['sigma_mu_mean_minus'] = syste['sigma_mu_mean'] - \
                    syste['syste']
                idxs = syste['zcomp'] == dbName
                selsyst = syste[idxs].to_records(index=False)
                nsn_tot = np.sum(selsyst['sigma_mu_mean'])
                nsn_tot_plus = np.sum(selsyst['sigma_mu_mean_plus'])
                nsn_tot_minus = np.sum(selsyst['sigma_mu_mean_minus'])
                print(zcomp, nsn_tot, nsn_tot_plus,
                      nsn_tot_minus, nsn_tot-nsn_tot_plus)
                ax.fill_between(
                    selsyst['z'], selsyst['sigma_mu_mean_plus'], selsyst['sigma_mu_mean_minus'], color='yellow')

        ax.grid()
        ax.set_ylabel('<$\sigma_\mu$>')
        ax.set_xlabel('$z$')
        ax.legend()
        ax.set_xlim([0.05, xmax])
        ax.set_ylim([0.0, None])

    def estimate_syste(self, data, data_syste):

        syste = {}
        data['zcomp'] = data['dbName']
        # print(data)
        syste_posl = []
        syste_negl = []
        for key, sel in data_syste.items():
            print('allo syste', key)
            sel['zcomp'] = sel['dbName']
            # print(sel)
            tt = data.merge(
                sel, left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            tt['delta_sigma_mu'] = tt['sigma_mu_mean_x']-tt['sigma_mu_mean_y']
            res = tt[['z', 'zcomp', 'delta_sigma_mu', 'sigma_mu_mean_x']]
            res = res.rename(columns={'sigma_mu_mean_x': 'sigma_mu_mean'})
            mean_delta = np.mean(res['delta_sigma_mu'])
            print(res)
            if mean_delta > 0:
                syste_posl.append(res)
            else:
                syste_negl.append(res)

        # now estimate positive and negative systematics

        if len(syste_posl) > 1:
            syste_pos = syste_posl[0].merge(
                syste_posl[1], left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            syste_pos['syste'] = np.sqrt(
                syste_pos['delta_sigma_mu_x']**2+syste_pos['delta_sigma_mu_y']**2)
            syste_pos['sigma_mu_syste'] = syste_pos['sigma_mu_mean_x'] + \
                syste_pos['syste']
            syste_pos['syste_type'] = 'pos'
            syste_pos = syste_pos.rename(
                columns={'sigma_mu_mean_x': 'sigma_mu_mean'})

        else:
            syste_pos = syste_posl[0]
            syste_pos['syste'] = np.sqrt(
                syste_pos['delta_sigma_mu']**2)
            syste_pos['sigma_mu_syste'] = syste_pos['sigma_mu_mean'] + \
                syste_pos['syste']
            syste_pos['syste_type'] = 'pos'

        if len(syste_negl) > 1:
            syste_neg = syste_negl[0].merge(
                syste_negl[1], left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            syste_neg['syste'] = np.sqrt(
                syste_neg['delta_sigma_mu_x']**2+syste_neg['delta_sigma_mu_y']**2)
            syste_neg['sigma_mu_syste'] = syste_neg['sigma_mu_mean_x'] - \
                syste_neg['syste']
            syste_neg['syste_type'] = 'neg'

            syste_neg = syste_neg.rename(
                columns={'sigma_mu_mean_x': 'sigma_mu_mean'})
        else:
            syste_neg = syste_negl[0]
            syste_neg['syste'] = np.sqrt(
                syste_neg['delta_sigma_mu']**2)
            syste_neg['sigma_mu_syste'] = syste_neg['sigma_mu_mean'] - \
                syste_neg['syste']
            syste_neg['syste_type'] = 'neg'

        vvar = ['z', 'zcomp', 'sigma_mu_mean', 'syste']

        syste = syste_pos[vvar].merge(syste_neg[vvar], left_on=[
                                      'z', 'zcomp'], right_on=['z', 'zcomp'])
        syste['syste'] = syste[['syste_x', 'syste_y']].max(axis=1)
        syste = syste.rename(columns={'sigma_mu_mean_x': 'sigma_mu_mean'})

        return syste


class Plot_NSN:
    """
    class to plot nsn vs z for a set of redshift completeness survey

    Parameters
    ----------------
    nsn_bias: list(str)
      name of the files to process (plot)
   nsn_bias_syste: list(str)
      name of the files for syste estimation (plot)
    zcomp: list(str), opt
      list of redshift completeness survey to plot (default: ['0.90', '0.80', '0.70', '0.65'])
    fieldNames: list(str), opt
     list of fields to plot (default: ['COSMOS', 'XMM-LSS', 'CDFS', 'ELAIS', 'ADFS'])
    """

    def __init__(self, nsn_bias, nsn_bias_syste,
                 zcomps=['0.90', '0.80', '0.70', '0.65'],
                 fieldNames=['COSMOS', 'XMM-LSS', 'CDFS', 'ELAIS', 'ADFS']):

        data = {}
        data_syste = {}

        for fichName in nsn_bias:
            data[fichName] = pd.read_hdf('{}.hdf5'.format(fichName))

        for fichName in nsn_bias_syste:
            data_syste[fichName] = pd.read_hdf('{}.hdf5'.format(fichName))

        ls = dict(
            zip(zcomps, ['solid', 'dotted', 'dashed', 'dashdot']))

        for field in fieldNames:
            fig, ax = plt.subplots(figsize=(12, 9))
            fig.suptitle('{}'.format(field))
            for key, vals in data.items():
                idx = vals['fieldName'] == field
                sel = vals[idx]
                syste = self.estimate_syste(field, sel, data_syste)
                self.plot_field(ax, field, sel, zcomps, ls, syste)

    def plot_field(self, ax, field, sel, zcomps, ls, syste):
        """
        This is where the plot is effectively made

        Parameters
        ---------------
        field: str
          name of the field to plot
        sel: pandas df
          data corresponding to this field
        zcomps: list(str)
          list of redshift completeness to draw
        ls: str
          linestyle for the plot

        """
        # fig, ax = plt.subplots(figsize=(12, 9))
        # fig.suptitle('{}'.format(field))
        print(type(sel))
        sel = sel.fillna(0)
        for zcomp in zcomps:
            idxb = sel['zcomp'] == zcomp
            selb = sel[idxb].to_records(index=False)
            idz = selb['z'] <= 1.09
            selb = selb[idz]
            """
            ax.plot(selb['z'], selb['nsn_eff'],
                    label='$z_{complete}$'+'= {}'.format(zcomp), lw=3, ls=ls[zcomp])
            """
            if not syste.empty:
                syste['nsn_eff_plus'] = syste['nsn_eff']+syste['syste']
                syste['nsn_eff_minus'] = syste['nsn_eff']-syste['syste']

                idxs = syste['zcomp'] == zcomp
                selsyst = syste[idxs].to_records(index=False)
                nsn_tot = np.sum(selsyst['nsn_eff'])
                nsn_tot_plus = np.sum(selsyst['nsn_eff_plus'])
                nsn_tot_minus = np.sum(selsyst['nsn_eff_minus'])
                print(zcomp, nsn_tot, nsn_tot_plus,
                      nsn_tot_minus, nsn_tot-nsn_tot_plus)
                ax.fill_between(
                    selsyst['z'], selsyst['nsn_eff_plus'], selsyst['nsn_eff_minus'], color='yellow')
                """
                xnew, smootha = self.smooth_it(
                    selsyst, vary='nsn_eff_plus', k=25)
                xnew, smoothb = self.smooth_it(
                    selsyst, vary='nsn_eff_minus', k=25)
                
                
                ax.fill_between(
                    xnew, smootha, smoothb, color='yellow')
                """
                # ax.plot(selsyst['z'], selsyst['nsn_syste'], lw=3, ls=ls[zcomp])

            xmax = np.max(selb['z'])
            """
            xnew = np.linspace(
                np.min(selb['z']), np.max(selb['z']), 100)
            spl = make_interp_spline(
                selb['z'], selb['nsn_eff'], k=7)  # type: BSpline
            print('NSN', zcomp, np.sum(
                selb['nsn_eff']), np.sum(selb['nsn_eff']))
            spl_smooth = spl(xnew)
            """
            xnew, spl_smooth = self.smooth_it(selb)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[zcomp], lw=3)

        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('N$_{\mathrm{SN}}$')
        ax.legend()
        ax.set_xlim([0.05, xmax])
        ax.set_ylim([0.0, None])

    def smooth_it(self, selb, varx='z', vary='nsn_eff', k=7):

        xmax = np.max(selb['z'])

        xnew = np.linspace(
            np.min(selb[varx]), np.max(selb[varx]), 100)
        spl = make_interp_spline(
            selb[varx], selb[vary], k=k)  # type: BSpline
        spl_smooth = spl(xnew)

        return xnew, spl_smooth

    def estimate_syste(self, field, data, data_syste):

        syste = {}

        # print(data)
        syste_posl = []
        syste_negl = []
        for key, vals in data_syste.items():
            print('allo syste', key)
            idx = vals['fieldName'] == field
            sel = vals[idx]
            # print(sel)
            tt = data.merge(
                sel, left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            tt['delta_n'] = tt['nsn_eff_x']-tt['nsn_eff_y']
            res = tt[['z', 'zcomp', 'delta_n', 'nsn_eff_x']]
            res = res.rename(columns={'nsn_eff_x': 'nsn_eff'})
            mean_delta = np.mean(res['delta_n'])
            print(res)
            if mean_delta > 0:
                syste_posl.append(res)
            else:
                syste_negl.append(res)

        # now estimate positive and negative systematics

        if len(syste_posl) > 1:
            syste_pos = syste_posl[0].merge(
                syste_posl[1], left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            syste_pos['syste'] = np.sqrt(
                syste_pos['delta_n_x']**2+syste_pos['delta_n_y']**2)
            syste_pos['nsn_syste'] = syste_pos['nsn_eff_x']+syste_pos['syste']
            syste_pos['syste_type'] = 'pos'
            syste_pos = syste_pos.rename(columns={'nsn_eff_x': 'nsn_eff'})

        else:
            syste_pos = syste_posl[0]
            syste_pos['syste'] = np.sqrt(
                syste_pos['delta_n']**2)
            syste_pos['nsn_syste'] = syste_pos['nsn_eff']+syste_pos['syste']
            syste_pos['syste_type'] = 'pos'

        if len(syste_negl) > 1:
            syste_neg = syste_negl[0].merge(
                syste_negl[1], left_on=['z', 'zcomp'], right_on=['z', 'zcomp'])
            syste_neg['syste'] = np.sqrt(
                syste_neg['delta_n_x']**2+syste_neg['delta_n_y']**2)
            syste_neg['nsn_syste'] = syste_neg['nsn_eff_x']-syste_neg['syste']
            syste_neg['syste_type'] = 'neg'

            syste_neg = syste_neg.rename(columns={'nsn_eff_x': 'nsn_eff'})
        else:
            syste_neg = syste_negl[0]
            syste_neg['syste'] = np.sqrt(
                syste_neg['delta_n']**2)
            syste_neg['nsn_syste'] = syste_neg['nsn_eff']-syste_neg['syste']
            syste_neg['syste_type'] = 'neg'

        vvar = ['z', 'zcomp', 'nsn_eff', 'syste']

        syste = syste_pos[vvar].merge(syste_neg[vvar], left_on=[
                                      'z', 'zcomp'], right_on=['z', 'zcomp'])
        syste['syste'] = syste[['syste_x', 'syste_y']].max(axis=1)
        syste = syste.rename(columns={'nsn_eff_x': 'nsn_eff'})

        return syste


parser = OptionParser(
    description='plot nsn and sigma_mu distributions')
parser.add_option("--sigma_mu", type=str, default='sigma_mu_from_simu_Ny_40',
                  help="sigma_mu file name [%default]")
parser.add_option("--nsn_bias", type=str, default='nsn_bias_Ny_40',
                  help="nsn bias file name [%default]")
parser.add_option("--nsn_bias_syste", type=str,
                  default='nsn_bias_Ny_40_color_plus_1_sigma,nsn_bias_Ny_40_color_minus_1_sigma,nsn_bias_Ny_40_x1_plus_1_sigma,nsn_bias_Ny_40_x1_minus_1_sigma,nsn_bias_Ny_40_mb_plus_1_sigma,nsn_bias_Ny_40_mb_minus_1_sigma', help="nsn bias syste files [%default]")
parser.add_option("--sigma_mu_syste", type=str,
                  default='sigma_mu_from_simu_Ny_40_mb_plus_1_sigma,sigma_mu_from_simu_Ny_40_mb_minus_1_sigma,sigma_mu_from_simu_Ny_40_x1_plus_1_sigma,sigma_mu_from_simu_Ny_40_x1_minus_1_sigma,sigma_mu_from_simu_Ny_40_color_plus_1_sigma,sigma_mu_from_simu_Ny_40_color_minus_1_sigma', help="nsn bias syste files [%default]")


opts, args = parser.parse_args()

nsn_bias = opts.nsn_bias.split(',')
nsn_bias_syste = opts.nsn_bias_syste.split(',')
sigma_mu = opts.sigma_mu.split(',')
sigma_mu_syste = opts.sigma_mu_syste.split(',')

Plot_NSN(nsn_bias, nsn_bias_syste, fieldNames=['CDFS'])
Plot_Sigma_mu(sigma_mu, sigma_mu_syste)
# Plot_Sigma_Components(sigma_mu, dbNames=['DD_0.90', 'DD_0.65'])
plt.show()
# save sigma_mb vs z
# get_sigma_mb_z(sigma_mu)
