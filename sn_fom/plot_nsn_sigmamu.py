import pandas as pd
from sn_fom import plt
import numpy as np
from scipy.interpolate import make_interp_spline
from optparse import OptionParser


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

    def __init__(self, fichName, dbNames=['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65']):

        self.dbNames = dbNames
        data = pd.read_hdf(fichName)
        self.plot_sigma_mu(data)

    def plot_sigma_mu(self, dd):
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

        ax.grid()
        ax.set_ylabel('<$\sigma_\mu$>')
        ax.set_xlabel('$z$')
        ax.legend()
        ax.set_xlim([0.05, xmax])
        ax.set_ylim([0.0, None])


class Plot_NSN:
    """
    class to plot nsn vs z for a set of redshift completeness survey

    Parameters
    ----------------
    fichname: str
      name of the file to process (plot)
    zcomp: list(str), opt
      list of redshift completeness survey to plot (default: ['0.90', '0.80', '0.70', '0.65'])
    fieldNames: list(str), opt
     list of fields to plot (default: ['COSMOS', 'XMM-LSS', 'CDFS', 'ELAIS', 'ADFS'])
    """

    def __init__(self, fichName,
                 zcomps=['0.90', '0.80', '0.70', '0.65'],
                 fieldNames=['COSMOS', 'XMM-LSS', 'CDFS', 'ELAIS', 'ADFS']):

        data = pd.read_hdf(fichName)
        ls = dict(
            zip(zcomps, ['solid', 'dotted', 'dashed', 'dashdot']))

        for field in fieldNames:
            idx = data['fieldName'] == field
            sel = data[idx]
            self.plot_field(field, sel, zcomps, ls)

    def plot_field(self, field, sel, zcomps, ls):
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
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.suptitle('{}'.format(field))
        print(type(sel))
        sel = sel.fillna(0)
        for zcomp in zcomps:
            idxb = sel['zcomp'] == zcomp
            selb = sel[idxb].to_records(index=False)
            idz = selb['z'] <= 1.09
            selb = selb[idz]
            """
            ax.plot(selb['z'], selb['nsn_eff'],
                    label='$z_{complee}$'+'= {}'.format(zcomp), lw=3)
            """
            xmax = np.max(selb['z'])
            xnew = np.linspace(
                np.min(selb['z']), np.max(selb['z']), 100)
            spl = make_interp_spline(
                selb['z'], selb['nsn_eff'], k=7)  # type: BSpline
            print('NSN', zcomp, np.sum(
                selb['nsn_eff']), np.sum(selb['nsn_eff']))
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[zcomp], lw=3)

        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('N$_{\mathrm{SN}}$')
        ax.legend()
        ax.set_xlim([0.05, xmax])
        ax.set_ylim([0.0, None])


parser = OptionParser(
    description='plot nsn and sigma_mu distributions')
parser.add_option("--Ny", type=str, default='40',
                  help="y-band visits [%default]")

opts, args = parser.parse_args()
Ny = opts.Ny

Plot_NSN('nsn_bias_Ny_{}.hdf5'.format(Ny), fieldNames=['CDFS'])
Plot_Sigma_mu('sigma_mu_from_simu_Ny_{}.hdf5'.format(Ny))
plt.show()
