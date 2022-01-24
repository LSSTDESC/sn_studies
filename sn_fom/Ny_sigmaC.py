from sn_fom.utils import loadSN, binned_data
import matplotlib.pyplot as plt
import numpy as np
import glob
from sn_tools.sn_io import loopStack
from sn_tools.sn_utils import multiproc
from astropy.table import Table
import pandas as pd
import os
from scipy.interpolate import interp1d, make_interp_spline


class SN_SNR:
    """
    class to add SNR Lc values to SN fitted parameters table

    Parameters
    --------------
    fitDir: str
      directory with fitted SN values
    simuDir: str
      directory with LC simulations
    dbName: str
       tag for prod
    snType: str, opt
      type of SN to consider (default: mediumSN)


    """

    def __init__(self, fitDir, simuDir, dbName, snType='mediumSN'):

        # get all the files related to fitted values
        search_path = '{}/{}/*{}*.hdf5'.format(fDir, dbName, snType)
        self.fitFiles = glob.glob(search_path)

        self.simuDir = simuDir
        self.dbName = dbName
        self.snType = snType
        self.outputDir = '{}/{}_SNR'.format(fitDir, dbName)
        # create config dir if necessary
        if not os.path.exists(self.outputDir):
            os.mkdir(self.outputDir)

    def __call__(self):

        for fi in self.fitFiles:
            fitName = fi.split('/')[-1]
            simuName = fitName.replace('Fit', 'Simu')
            lcName = fitName.replace('Fit', 'LC')

            path_simu = '{}/{}/{}'.format(self.simuDir,
                                          self.dbName, simuName).replace('_sn_cosmo', '')
            path_lc = '{}/{}/{}'.format(self.simuDir, self.dbName,
                                        lcName).replace('_sn_cosmo', '')

            print(path_simu, path_lc)
            self.loop_SNR(fitName, fi, path_simu, path_lc)

    def loop_SNR(self, fitName, path_fit, path_simu, path_lc):

        data_fit = loopStack([path_fit], objtype='astropyTable').to_pandas()
        data_simu = loopStack([path_simu], objtype='astropyTable').to_pandas()

        data_fit['SNID'] = data_fit['SNID'].str.decode('utf-8')
        data_simu['SNID'] = data_simu['SNID'].str.decode('utf-8')
        data_simu['index_hdf5'] = data_simu['index_hdf5'].str.decode('utf-8')

        df_snr = pd.DataFrame()
        for io, row in data_fit.iterrows():
            # get the simu infos
            idx = data_simu['SNID'] == row['SNID']
            index_hdf5 = data_simu[idx]['index_hdf5'].values[0]
            # grab LC
            snr_lc = self.get_SNR(path_lc, index_hdf5, row['SNID'])
            df_snr = pd.concat((df_snr, snr_lc))

        df_res = data_fit.merge(df_snr, left_on=['SNID'], right_on=['SNID'])

        # save the file here
        outName = '{}/{}'.format(self.outputDir, fitName)
        Table.from_pandas(df_res).write(outName, 'fit_SNR',
                                        compression=True, overwrite=True)

    def get_SNR(self, path_lc, index_hdf5, SNID):

        lc = Table.read(path_lc, path='lc_{}'.format(index_hdf5))

        df = pd.DataFrame([SNID], columns=['SNID'])
        r = []
        for band in 'grizy':
            idx = lc['filter'] == band
            sel_lc = lc[idx]
            snr_band = 0.
            if len(sel_lc) > 0:
                sel_lc['SNR'] = sel_lc['flux']/sel_lc['fluxerr']
                snr_band = np.sqrt(np.sum(sel_lc['SNR']**2))
            df['SNR{}'.format(band)] = snr_band

        return df


def loadData(fDir, dbName, snType='mediumSN'):

    data = loadSN(fDir, dbName, snType)
    # data['fitstatus'] = data['fitstatus'].str.decode('utf-8')
    print('hhh', data['fitstatus'])
    idx = data['fitstatus'] == 'fitok'
    data = data[idx]
    data['sigmaC'] = np.sqrt(data['Cov_colorcolor'])

    return data


def loadData_new(dbNames, params, j=0, output_q=None):

    params['fDir'] = fDir
    params['snType'] = snType

    data_res = {}
    for dbName in dbNames:
        data = loadSN(fDir, dbName, snType)
        # data['fitstatus'] = data['fitstatus'].str.decode('utf-8')
        print('hhh', data['fitstatus'])
        idx = data['fitstatus'] == 'fitok'
        data = data[idx]
        data['sigmaC'] = np.sqrt(data['Cov_colorcolor'])
        data_res[dbName] = data

    if output_q is not None:
        return output_q.put({j: data_res})
    else:
        return data_res


class Plot:
    def __init__(self, data, vars=['sigmaC', 'SNRi', 'SNRz', 'SNRy']):

        self.vars = vars

        dict_bin = {}
        dict_bin['z'] = {}
        dict_bin['z']['min'] = 0.01
        dict_bin['z']['max'] = 1.10
        dict_bin['z']['nbins'] = 60

        dict_bin['sigmaC'] = {}
        dict_bin['sigmaC']['min'] = 0.01
        dict_bin['sigmaC']['max'] = 0.06
        dict_bin['sigmaC']['nbins'] = 20
        binvars = ['z']+['sigmaC']*3
        self.bbins = dict(zip(vars, binvars))
        self.dict_bin = dict_bin

        df_res = pd.DataFrame()
        for key, vals in data.items():
            vvals = key.split('_')
            configName = 'config_fakes_{}_{}.csv'.format(
                vvals[-2], vvals[-1])
            print(configName)
            df = pd.read_csv(configName)
            idx = df['dbName'] == key
            row = pd.DataFrame(df[idx])
            print(row)
            zlim, zlimp, zlimm = self.get_zlim(vals)
            row['zlim'] = zlim
            row['zlimp'] = zlimp
            row['zlimm'] = zlimm

            print(zlim)
            df_res = pd.concat((df_res, row))

        print(df_res)
        self.plot_zlim(df_res)

        plot = False

        if plot:
            self.plot_all(data)

    def plot_zlim(self, data):

        data = data.sort_values(by=['Nvisits_y'])
        data['DD_zcomp'] = data['dbName'].str.split(
            '_', expand=True)[0]+'_'+data['dbName'].str.split('_', expand=True)[1]
        data['zcomp'] = data['DD_zcomp'].str.split('_', expand=True)[1]
        print(data)
        fig, ax = plt.subplots(figsize=(15, 8))
        xvar = 'Nvisits_y'
        yvar = 'zlim'
        Nycut = dict(zip(['DD_0.70', 'DD_0.80', 'DD_0.90',
                          'DD_0.75', 'DD_0.85'], [50, 90, 90, 90, 90]))
        ls = dict(zip(['DD_0.70', 'DD_0.80', 'DD_0.90', 'DD_0.75', 'DD_0.85'],
                      ['solid', 'dashed', 'dotted', (0, (1, 10)), (0, (3, 5, 1, 5))]))
        for confname in data['DD_zcomp'].unique():
            idx = data['DD_zcomp'] == confname
            idx &= data['Nvisits_y'] <= Nycut[confname]
            seldf = data[idx]
            zcomp = float(seldf['zcomp'].unique()[0])
            seldf = seldf.to_records(index=False)
            ax.plot(seldf[xvar], seldf[yvar], ls=ls[confname],
                    color='k', label='$z_{complete}=$'+'{}'.format(zcomp), lw=3)

            """
            xmin, xmax = np.min(seldf[xvar]), np.max(seldf[xvar])
            #ip = interp1d(seldf[xvar],seldf[yvar],bounds_error=False, fill_value=0.)

            xnew = np.linspace(xmin, xmax, 70)
            spl = make_interp_spline(
                seldf[xvar], seldf[yvar], k=3)  # type: BSpline
            spl_smooth = spl(xnew)
            ax.plot(xnew, spl_smooth,
                    label='$z_{complete}$'+'= {}'.format(zcomp), lw=3)
            """
            """
            ax.fill_between(
                seldf[xvar], seldf['{}p'.format(yvar)], seldf['{}m'.format(yvar)], color='yellow')
            """
            ax.fill_between(
                seldf[xvar], seldf[yvar]+0.01, seldf[yvar]-0.01, color='yellow')
        ax.grid()
        ax.set_xlabel('$Y$-band visits')
        ax.set_ylabel('$z_{limit}$')
        ax.legend()

    def plot_all(self, data):

        for vv in vars:
            fig, ax = plt.subplots()
            r = []
            for key, vals in data.items():
                if 'SNRi' in vals.columns:
                    vals['SNR'] = 0
                    for b in 'grizy':
                        vals['SNR'] += vals['SNR{}'.format(b)]
                    vals['SNR'] = vals['SNR']
                    for b in 'grizy':
                        vals['SNR{}'.format(
                            b)] = vals['SNR{}'.format(b)]/vals['SNR']
                self.plot(vals)

    def get_zlim(self, data, var='sigmaC', sigmaC=0.04):

        dd = self.bbins[var]
        vmin = self.dict_bin[dd]['min']
        vmax = self.dict_bin[dd]['max']
        nbins = self.dict_bin[dd]['nbins']
        bdata = binned_data(vmin, vmax, nbins, data, dd, var)

        # get the redshift limit here
        bb = interp1d(
            data[var], data[dd], bounds_error=False, fill_value=0.)
        return bb(sigmaC), bb(1.05*sigmaC), bb(0.95*sigmaC)

    def plot(self, ax, data, var, label):

        dd = self.bbins[var]
        vmin = self.dict_bin[dd]['min']
        vmax = self.dict_bin[dd]['max']
        nbins = self.dict_bin[dd]['nbins']
        bdata = binned_data(vmin, vmax, nbins, data, dd, var)
        ax.plot(bdata[dd], bdata['{}_mean'.format(var)], label=label)
        ax.grid()
        ax.legend()

    """
    ax[0, 0].legend()
    ax[0, 0].set_ylim([0.0, 0.05])
    ax[1, 0].set_xlim([0.01, 0.06])
    ax[1, 1].set_xlim([0.01, 0.06])
    ax[0, 1].set_xlim([0.01, 0.06])

    ax[0].legend()
    ax[0].set_ylim([0.0, 0.05])
    """


snType = 'faintSN'
snType = 'mediumSN'
Ny = list(range(10, 90, 10))
# Ny += [80]

fDir = 'Fakes_medium/Fit'
"""
simuDir = 'Fakes_medium/Simu'
for Nyv in Ny:
    dbName = 'DD_0.70_Ny_{}'.format(Nyv)
    ana = SN_SNR(fDir, simuDir, dbName, snType=snType)
    ana()
"""

dbNames = []
add_vv = '_SNR'
add_vv = ''
for zcomp in ['0.70', '0.75', '0.80']:
    # for zcomp in ['0.70', '0.80', '0.90', '0.75', '0.85']:
    for Nyv in Ny:
        dbNames.append('DD_{}_Ny_{}a{}'.format(zcomp, Nyv, add_vv))

dbNames.append('DD_0.70_Ny_40')
data = {}

params = {}
params['fDir'] = fDir
params['snType'] = snType
data = multiproc(dbNames, params, loadData_new, 4)

"""
for dbName in dbNames:
    data[dbName] = loadData(fDir, dbName, snType=snType)
    print(data[dbName]['SNID'])
"""
Plot(data, vars=['sigmaC'])
# Plot(data)

plt.show()
