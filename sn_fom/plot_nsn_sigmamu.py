import pandas as pd
from sn_fom import plt
import numpy as np
from scipy.interpolate import make_interp_spline


class Plot_Sigma_mu:

    def __init__(self, fichName, dbNames=['DD_0.90', 'DD_0.80', 'DD_0.70', 'DD_0.65']):

        self.dbNames = dbNames
        data = pd.read_hdf(fichName)
        self.plot_sigma_mu(data)

    def plot_sigma_mu(self, dd):
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
            zcomp = dbName.split('_')[-1]
            ax.plot(sel['z'], sel['sigma_mu_mean'],
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)
            """
            xnew = np.linspace(
                np.min(sel['z']), np.max(sel['z']), 100)
            spl = make_interp_spline(
                sel['z'], sel['sigma_mu_mean'], k=9)  # type: BSpline
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[dbName], lw=3)
            """
        ax.grid()
        ax.set_ylabel('<$\sigma_\mu$>')
        ax.set_xlabel('$z$')
        ax.legend()
        #ax.set_xlim([0.05, 1.075])
        ax.set_ylim([0.0, None])


class Plot_NSN:

    def __init__(self, fichName,
                 zcomps=['0.90', '0.80', '0.70', '0.65'],
                 fieldNames=['COSMOS', 'XMM-LSS', 'CDFS', 'ELAIS', 'ADFS']):
        """
        Method to plot nsn vs z for each field and config

        """
        data = pd.read_hdf(fichName)
        ls = dict(
            zip(zcomps, ['solid', 'dotted', 'dashed', 'dashdot']))

        for field in fieldNames:
            idx = data['fieldName'] == field
            sel = data[idx]
            self.plot_field(field, sel, zcomps, ls)

    def plot_field(self, field, sel, zcomps, ls):

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.suptitle('{}'.format(field))
        print(type(sel))
        sel = sel.fillna(0)
        for zcomp in zcomps:
            idxb = sel['zcomp'] == zcomp
            selb = sel[idxb].to_records(index=False)
            idz = selb['z'] >= 0.9
            selbz = selb[idz]
            # ax.plot(selb['z'], selb['nsn_eff'],
            #        label='$z_{complee}$'+'= {}'.format(zcomp))
            xnew = np.linspace(
                np.min(selb['z']), np.max(selb['z']), 100)
            spl = make_interp_spline(
                selb['z'], selb['nsn_eff'], k=11)  # type: BSpline
            print('NSN', zcomp, np.sum(
                selb['nsn_eff']), np.sum(selbz['nsn_eff']))
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), ls=ls[zcomp], lw=3)
        ax.grid()
        ax.set_xlabel('$z$')
        ax.set_ylabel('N$_{\mathrm{SN}}$')
        ax.legend()
        ax.set_xlim([0.05, 1.12])
        ax.set_ylim([0.0, None])


#Plot_NSN('nsn_bias_Ny_80.hdf5', fieldNames=['CDFS'])
Plot_Sigma_mu('sigma_mu_from_simu_Ny_120.hdf5')
plt.show()
