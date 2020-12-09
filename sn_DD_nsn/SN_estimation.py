import glob
from sn_tools.sn_io import loopStack
from astropy.table import Table, vstack
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sn_tools.sn_rate import SN_Rate


class Effi_SN:
    """
    class to estimate observing efficiencies vs redshift

    Parameters
    ---------------
    sn_tot: pandas df
      data to process

    """

    def __init__(self, sn_tot, min_rf_phase_qual=-15., max_rf_phase_qual=25.):

        self.rateSN = SN_Rate(
            min_rf_phase=min_rf_phase_qual, max_rf_phase=max_rf_phase_qual)
        self.effi = self.effi_calc(sn_tot)

    def effi_calc(self, sn_tot, color_cut=0.04):
        """
        Method estimating efficiency vs z for a sigma_color cut
        Parameters
        ---------------
        sn_tot: pandas df
         data used to estimate efficiencies
        color_cut: float, opt
         color selection cut (default: 0.04)

        Returns
        ----------
        effi: pandas df with the following cols:
        season: season
        pixRA: RA of the pixel
        pixDec: Dec of the pixel
        healpixID: pixel ID
        x1: SN stretch
        color: SN color
        z: redshift
        effi: efficiency
        effi_err: efficiency error (binomial)
        """

        sndf = pd.DataFrame(sn_tot)

        print(sndf.columns)
        listNames = ['season', 'pixRA', 'pixDec', 'healpixID', 'x1', 'color']
        groups = sndf.groupby(listNames)

        # estimating efficiencies
        effi = groups[['Cov_colorcolor', 'z']].apply(
            lambda x: self.effiObsdf(x, color_cut)).reset_index(level=list(range(len(listNames))))

        print('effi', effi)
        """
        # this is to plot efficiencies and also sigma_color vs z
        if ploteffi:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots()
            # figb, axb = plt.subplots()

            plot(ax, effi, 'effi', 'effi_err',
                      'Observing Efficiencies', ls='-')
            # sndf['sigma_color'] = np.sqrt(sndf['Cov_colorcolor'])
            # self.plot(axb, sndf, 'sigma_color', None, '$\sigma_{color}$')

            plt.show()
        """
        return effi

    def effiObsdf(self, data, color_cut=0.04):
        """
        Method to estimate observing efficiencies for supernovae
        Parameters
        --------------
        data: pandas df - grp
        data to process

        Returns
        ----------
        pandas df with the following cols:
        - cols used to make the group
        - effi, effi_err: observing efficiency and associated error
        """

        # reference df to estimate efficiencies
        df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]

        # selection on sigma_c<= 0.04
        df_sel = df.loc[lambda dfa:  np.sqrt(
            dfa['Cov_colorcolor']) <= color_cut, :]

        # make ratio using histogram
        x1 = df_sel['z'].values
        x2 = df['z'].values
        bins = np.array([0.01]+list(np.arange(0.025, 0.8, 0.05)))
        val_of_bins_x1, edges_of_bins_x1, patches_x1 = plt.hist(
            x1, bins=bins, histtype='step', label="x1")
        val_of_bins_x2, edges_of_bins_x2, patches_x2 = plt.hist(
            x2, bins=bins, histtype='step', label="x2")

        ratio = np.divide(val_of_bins_x1,
                          val_of_bins_x2,
                          where=(val_of_bins_x2 != 0))

        var = val_of_bins_x1*ratio*(1.-ratio)
        bins_centered = 0.5*(bins[1:]+bins[:-1])

        return pd.DataFrame({'z': bins_centered,
                             'effi': ratio,
                             'effi_err': np.sqrt(var),
                             'effi_var': var})

        print(test)
        # make groups (with z)
        group = df.groupby('z')
        group_sel = df_sel.groupby('z')

        # Take the ratio to get efficiencies
        rb = (group_sel.size()/group.size())
        err = np.sqrt(rb*(1.-rb)/group.size())
        var = rb*(1.-rb)*group.size()

        rb = rb.array
        err = err.array
        var = var.array

        rb[np.isnan(rb)] = 0.
        err[np.isnan(err)] = 0.
        var[np.isnan(var)] = 0.

        return pd.DataFrame({group.keys: list(group.groups.keys()),
                             'effi': rb,
                             'effi_err': err,
                             'effi_var': var})

    def ploteffi(self, effi):
        import matplotlib.pylab as plt
        # fig, ax = plt.subplots()
        # figb, axb = plt.subplots()

        print('boo', np.unique(effi['healpixID']))
        for healpixID in np.unique(effi['healpixID']):
            idx = effi['healpixID'] == healpixID
            sel = effi[idx]
            grp = sel.groupby(['x1', 'color', 'season']).apply(lambda x: self.plot(
                x, 'effi', 'effi_err', 'Observing Efficiencies', ls='-'))
            print('processing', healpixID)
            plt.show()

    def plot(self, grp, vary, erry=None, legy='', ls='None'):
        """
        Simple method to plot vs z
        Parameters
        --------------
        effi: pandas df
          data to plot
        vary: str
          variable (column of effi) to plot
        erry: str, opt
          error on y-axis (default: None)
        legy: str, opt
          y-axis legend (default: '')
        """

        fig, ax = plt.subplots()
        yerr = None

        x1 = grp['x1'].unique()[0]
        color = grp['color'].unique()[0]
        if erry is not None:
            yerr = grp[erry]
        ax.errorbar(grp['z'], grp[vary], yerr=yerr,
                    marker='o', label='(x1,color)=({},{})'.format(x1, color), lineStyle=ls)

        ftsize = 15
        ax.set_xlabel('z', fontsize=ftsize)
        ax.set_ylabel(legy, fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.legend(fontsize=ftsize)


def pixels(grp):

    return pd.DataFrame({'npixels': [len(np.unique(grp['healpixID']))]})


def finalNums(grp, normfact=10.):

    return pd.DataFrame({'nsn_zlim': [grp['nsn_zlim'].sum()/normfact],
                         'nsn_tot': [grp['nsn_tot'].sum()/normfact],
                         'zlim': [grp['zlim'].median()]})


def nSN(grp, sigmaC=0.04):

    idx = grp['Cov_colorcolor'] <= sigmaC**2
    sela = grp[idx]
    idx &= grp['z'] <= grp['zlim']
    sel = grp[idx]

    return pd.DataFrame({'nsn_zlim': [len(sel)],
                         'nsn_tot': [len(sela)],
                         'zlim': [grp['zlim'].median()]})


def SN(fitDir, dbName, fieldNames, SNtype):

    dfSN = pd.DataFrame()
    for field in fieldNames:
        search_path = '{}/{}/*{}*{}*.hdf5'.format(
            fitDir, dbName, field, SNtype)
        fis = glob.glob(search_path)[:20]
        print('aooou', fis, search_path)
        out = loopStack(fis, objtype='astropyTable').to_pandas()
        out['fieldName'] = field
        idx = out['Cov_colorcolor'] >= 1.e-5
        dfSN = pd.concat((dfSN, out[idx]))

    return dfSN


def zlim(grp, sigmaC=0.04):

    ic = grp['Cov_colorcolor'] <= sigmaC**2
    selb = grp[ic]

    if len(selb) == 0:
        zl = 0.
    else:
        zl = np.max(selb['z'])

    return pd.DataFrame({'zlim': [zl]})


mainDir = '/media/philippe/LSSTStorage/DD_new'
mainDir = '/home/philippe/LSST/DD_Full_Simu_Fit'
fitDir = '{}/Fit'.format(mainDir)
# fitDir = 'OutputFit'
simuDir = '{}/Simu'.format(mainDir)

dbName = 'descddf_v1.5_10yrs'
fieldNames = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAIS', 'ADFS1', 'ADFS2']
fieldNames = ['COSMOS']

allSN = pd.DataFrame()
zlimit = None

# get faintSN

faintSN = SN(fitDir, dbName, fieldNames, 'faintSN')

eff_SN = Effi_SN(faintSN)

eff_SN.ploteffi(eff_SN.effi)


print(test)

zlimit = faintSN.groupby(
    ['healpixID', 'fieldName', 'season']).apply(lambda x: zlim(x))

print(np.median(zlimit['zlim']))

print(test)
allSN = SN(fitDir, dbName, fieldNames, 'allSN')
print(allSN.groupby(['fieldName', 'season']).apply(lambda x: pixels(x)))
allSN = allSN.merge(zlimit, left_on=['fieldName', 'season'], right_on=[
    'fieldName', 'season'])

sumSN = allSN.groupby(['healpixID', 'fieldName', 'season']
                      ).apply(lambda x: nSN(x)).reset_index()

print(sumSN.groupby(['fieldName', 'season']).apply(
    lambda x: finalNums(x)).reset_index())

print(sumSN.groupby(['fieldName']).apply(
    lambda x: finalNums(x)).reset_index())

print('Total number of SN', finalNums(sumSN))

"""
tab = out['fullSN']

for season in np.unique(tab['season']):
    idx = tab['season'] == season
    sel = tab[idx]
    ib = zlimit
plt.plot(tab['z'],np.sqrt(tab['Cov_colorcolor']),'ko')
plt.show()
"""
