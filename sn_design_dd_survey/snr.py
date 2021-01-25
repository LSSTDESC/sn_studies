import os
import pandas as pd
import numpy as np
import time
from scipy.spatial import distance
from scipy.interpolate import RegularGridInterpolator, interp1d
from astropy.table import Table, unique, vstack
import multiprocessing

from .utils import flux5_to_m5, srand, gamma
from .wrapper import Nvisits_cadence
from . import plt
from . import filtercolors
from sn_tools.sn_calcFast import CovColor


class SNR:
    def __init__(self, dirSNR, data,
                 SNR_par, SNR_m5_file, SNR_min=1.,
                 zref=[0.1],
                 error_model=1,
                 save_SNR_combi=False, verbose=False, nproc=8):
        """
        Wrapper class to estimate SNR

        Parameters
        --------------
        dirSNR: str
        directory where SNR files will be located
        data: pandas df
        data to process (LC)
        SNR_par: dict
        SNR_m5_file: str
           SNR vs m5 file name
        SNR_min: float
           min SNR for LC points
        zref: list(float)
          list of redshift values to consider
        error_model: int
          to tag for error_model or not
        SNR parameters
        verbose: str, opt
          verbose mode for debugging
        nproc: int,opt
          number of procs for multiprocessing (default: 8)

        """
        # get the data parameters
        # useful for the SNR filename
        self.x1 = data.x1
        self.color = data.color
        self.bands = data.bands
        self.blue_cutoff = data.blue_cutoff
        self.red_cutoff = data.red_cutoff
        self.error_model = error_model
        self.verbose = verbose

        myclass = SNR_z(dirSNR, data,
                        SNR_par=SNR_par,
                        SNR_m5_file=SNR_m5_file,
                        SNR_min=SNR_min,
                        zref=zref,
                        save_SNR_combi=save_SNR_combi,
                        verbose=self.verbose,
                        nproc=nproc)

        # dfsigmaC = myclass.sigmaC_SNR()
        SNR_dict = myclass.sigmaC_SNR()
        # myclass.plotSigmaC_SNR(dfsigmaC) # plot the results

        # plt.show()
        # now choose SNR corresponding to sigmaC~0.04 vs z

        print('resultat', SNR_dict.keys())
        if self.verbose:
            print('SNR class - sigma_C selection')
            # SNR_dict = myclass.SNR(dfsigmaC)

        if self.verbose:
            print('SNR class - saving SNR files')

        # SNR_par_dict = dict(
        #   zip(['max', 'step', 'choice'], [50., 2., 'Nvisits']))
        # thename = self.name(SNRDir, SNR_par)
        # save the file
        # np.save(thename, np.copy(SNR_dict.to_records(index=False)))

        """
        for key, vals in SNR_dict.items():
        SNR_par_dict = dict(
        zip(['max', 'step', 'choice'], [50., 2., key]))
        thename = self.name(SNRDir, SNR_par_dict)
        # save the file
        np.save(thename, np.copy(vals.to_records(index=False)))
        """

        # self.SNR = pd.DataFrame(np.load(SNRName, allow_pickle=True))
        # print('loading', self.SNR)

    def plot(self):
        """
        plt the results

        Parameters
        ---------

        Returns
        --------
        plt SNR-band vs z

        """

        plotSNR_z(self.SNR, x1=self.x1, color=self.color, bands=self.bands)

    def name(self, SNRDir,
             SNR_par):
        """
        method to define the name of the SNR file

        Parameters
        ---------
        SNRDir: str,
          location directory of the file
        SNR_par: dict
         SNR parameters

        Returns
        --------
        full path to SNR file (str)

        """

        cutoff = '{}_{}'.format(self.blue_cutoff, self.red_cutoff)
        if self.error_model:
            cutoff = 'error_model'
        name = '{}/SNR_{}_{}_{}_{}_{}_{}.npy'.format(SNRDir,
                                                     self.x1,
                                                     self.color,
                                                     SNR_par['step'],
                                                     cutoff,
                                                     self.bands,
                                                     SNR_par['choice'])

        return name


class SNR_z:

    def __init__(self, dirSNR, data,
                 SNR_par={},
                 SNR_m5_file='',
                 SNR_min=1.,
                 zref=[0.1],
                 sigma_color_cut=0.04,
                 save_SNR_combi=False,
                 verbose=False,
                 nproc=8):
        """
        class to estimate SNR per band vs redshift

        Parameters
        ---------
        dirSNR: str
         directory where SNR files will be located
        data: pandas df
         data to process (LC)
        SNR_par: dict
          SNR parameters
        SNR_m5_file: str, opt
          SNR vs m5 file name
        SNR_min: float
           min SNR for LC points
        zref: list(float)
          list of redshift values to consider
        sigma_color_cut: float, opt
          selection on sigma_color (default: 0.04)
        save_SNR_combi: bool, opt
          to save SNR combination for sigmaC estimation
        verbose: str, opt
          verbose mode for debugging
        nproc: int
         number of procs for multiprocessing

        """
        # verbose
        # verbose = True
        self.verbose = verbose
        # get data parameters
        self.x1 = data.x1
        self.color = data.color
        self.bands = data.bands
        self.sigma_color_cut = sigma_color_cut
        self.save_SNR_combi = save_SNR_combi
        self.nproc = nproc
        self.dirSNR = dirSNR

        # get SNR parameters
        self.SNR_par = SNR_par

        # SNR min for LC points
        self.SNR_min = SNR_min

        # this list is requested to estimate Fisher matrix elements

        self.listcol = ['F_x0x0', 'F_x0x1',
                        'F_x0daymax', 'F_x0color',
                        'F_x1x1', 'F_x1daymax',
                        'F_x1color', 'F_daymaxdaymax',
                        'F_daymaxcolor', 'F_colorcolor']

        # load LC

        data.lc = data.lc.round({'z': 2})
        idx = data.lc['fluxerr_model'] >= 0.
        idx &= data.lc['flux'] > 1.e-10
        idx &= data.lc['z'].isin(zref)

        self.lcdf = data.lc.loc[idx]

        # estimate the derivative vs Fisher parameters here
        for vv in self.listcol:
            self.lcdf.loc[:, 'd_{}'.format(
                vv.split('_')[-1])] = self.lcdf[vv]*self.lcdf['fluxerr']**2

        # add SNR from error_model
        self.lcdf.loc[:, 'SNR_model'] = self.lcdf['flux'] / \
            self.lcdf['fluxerr_model']

        # for index, val in self.lcdf.iterrows():
        #    print('lc', val['flux_e_sec'])

        # print(test)
        # load m5
        self.medm5 = data.m5_Band

        # get gammas
        self.gamma = gamma('grizy')

        # load signal fraction per band
        self.fracSignalBand = data.fracSignalBand.fracSignalBand

        # map flux5-> m5

        self.f5_to_m5 = flux5_to_m5(self.bands)

        # load SNR_m5 and make griddata
        snr_m5 = np.load(SNR_m5_file, allow_pickle=True)

        """
        self.m5_from_SNR = {}

        for b in np.unique(snr_m5['band']):
            idx = snr_m5['band'] == b
            sela = Table(snr_m5[idx])
            sela['z'] = sela['z'].data.round(decimals=2)
            sela['SNR'] = sela['SNR'].data.round(decimals=2)

            snr_min = np.round(np.min(sela['SNR']), 1)
            # snr_max = np.round(np.max(sela['SNR']), 1)
            snr_max = 200.0
            print('hello', snr_min, snr_max)
            snr_range = np.arange(snr_min, snr_max, 0.1)
            sel = Table()
            for vl in unique(sela, keys='z'):
                idx = np.abs(sela['z']-vl['z']) < 1.e-5
                val = sela[idx]
                interp = interp1d(val['SNR'], val['m5'],
                                  bounds_error=False, fill_value=0.)
                re = interp(snr_range)
                tabb = Table(names=['m5'])
                tabb['m5'] = re.tolist()
                tabb['SNR'] = snr_range.tolist()
                tabb['z'] = [vl['z']]*len(snr_range)
                sel = vstack([sel, tabb])

            sel['z'] = sel['z'].data.round(decimals=2)
            sel['SNR'] = sel['SNR'].data.round(decimals=2)
            sel['m5'] = sel['m5'].data.round(decimals=3)

            zmin, zmax, zstep, nz = self.limVals(sel, 'z')
            snrmin, snrmax, snrstep, nsnr = self.limVals(sel, 'SNR')
            m5min, m5max, m5step, nm5 = self.limVals(sel, 'm5')

            zv = np.linspace(zmin, zmax, nz)
            snrv = np.linspace(snrmin, snrmax, nsnr)
            m5v = np.linspace(m5min, m5max, nm5)

            print('hello', nsnr, nz, len(sel))
            index = np.lexsort((sel['z'], sel['SNR']))
            m5extra = np.reshape(sel[index]['m5'], (nsnr, nz))

            self.m5_from_SNR[b] = RegularGridInterpolator(
                (snrv, zv), m5extra, method='linear', bounds_error=False, fill_value=0.)

        print('test extrapo', self.m5_from_SNR['z'](([30.], [0.7])))
        """
        self.m5_from_SNR, self.snrdict = self.grid_z(snr_m5, minx=0.)

        self.SNR_from_m5, self.snrdictb = self.grid_z(
            snr_m5, whatx='m5', minx=20., maxx=30., whatstep=0.1, whatz='SNR')

    def grid_z(self, snr_m5, whatx='SNR', minx=1., maxx=200., whatstep=0.1, whatz='m5'):

        dict_extrapo = {}
        snrdict = {}

        for b in np.unique(snr_m5['band']):
            idx = snr_m5['band'] == b
            sela = Table(snr_m5[idx])
            sela['z'] = sela['z'].data.round(decimals=2)
            sela[whatx] = sela[whatx].data.round(decimals=4)
            snr_min = np.round(minx, 1)
            # snr_max = np.round(np.max(sela['SNR']), 1)
            snr_max = maxx

            snr_range = np.arange(snr_min, snr_max, whatstep)
            sel = Table()
            for vl in unique(sela, keys='z'):
                idx = np.abs(sela['z']-vl['z']) < 1.e-5
                val = sela[idx]
                if len(val) >= 2:
                    interp = interp1d(val[whatx], val[whatz],
                                      bounds_error=False, fill_value=0.)
                    re = interp(snr_range)
                    tabb = Table(names=[whatz])
                    tabb[whatz] = re.tolist()
                    tabb[whatx] = snr_range.tolist()
                    tabb['z'] = [vl['z']]*len(snr_range)
                    sel = vstack([sel, tabb])

            """
            sel['z'] = sel['z'].data.round(decimals=2)
            sel[whatx] = sel[whatx].data.round(decimals=2)
            sel[whatz] = sel[whatz].data.round(decimals=3)
            """
            zmin, zmax, zstep, nz = self.limVals(sel, 'z')
            snrmin, snrmax, snrstep, nsnr = self.limVals(sel, whatx)
            snrdict[b] = (snrmin, snrmax)

            """
            zstep = np.round(zstep, 3)
            snrstep = np.round(snrstep, 3)
            m5step = np.round(m5step, 3)
            """

            zv = np.linspace(zmin, zmax, nz)
            snrv = np.linspace(snrmin, snrmax, nsnr)

            index = np.lexsort((sel['z'], sel[whatx]))
            m5extra = np.reshape(sel[index][whatz], (nsnr, nz))

            dict_extrapo[b] = RegularGridInterpolator(
                (snrv, zv), m5extra, method='linear', bounds_error=False, fill_value=0.)

        # print('test extrapo 2', dict_extrapo['z'](([30.], [0.7])))

        return dict_extrapo, snrdict

    def limVals(self, lc, field):
        """ Get unique values of a field in  a table

        Parameters
        ----------
        lc: Table
         astropy Table (here probably a LC)
        field: str
         name of the field of interest
        Returns
        -------
        vmin: float
         min value of the field
        vmax: float
         max value of the field
        vstep: float
         step value for this field (median)
        nvals: int
         number of unique values
        """

        lc.sort(field)
        vals = np.unique(lc[field].data.round(decimals=4))
        # print(vals)
        vmin = np.min(vals)
        vmax = np.max(vals)
        vstep = np.median(vals[1:]-vals[:-1])

        return vmin, vmax, vstep, len(vals)

    def sumgrp(self, grp):
        """
        Method to assess the sum of the grp column

        Parameters
        ----------
        grp: group (pandas df)

        Returns
        --------
        panda df with the following columns:
        - sumflux = sqrt(sum(flux**2))
        - flux5 = mean(flux_5)
        - SNR = 5*sumflux/flux_5
        - self.listcol = sum of Fisher elements


        """
        sumli = grp[self.listcol].sum()
        sumflux = np.sqrt(np.sum(grp['flux_e_sec']**2.))
        SNR = 5.*sumflux/np.mean(grp['flux_5'])

        dicta = {}

        for ll in self.listcol:
            dicta[ll] = [sumli[ll]]

        dicta['sumflux'] = [np.sqrt(np.sum(grp['flux_e_sec']**2.))]
        dicta['SNR'] = [5.*sumflux/np.mean(grp['flux_5'])]
        dicta['flux_5'] = np.mean(grp['flux_5'])

        return pd.DataFrame.from_dict(dicta)

    def sigmaC_SNR(self):
        """
        Method to assess sigma_C as a function of SNRbands

        Parameters
        ---------------

        Returns
        -----------
        pandas df with the following columns:
        x1,color,z,band
        (sum_flux,SNR,self.listcol,flux_5_e_sec,m5_calc,flux_5)_band
        sigmaC,SNRcalc_tot


        """

        if self.verbose:
            print('SNR_z - sigmaC_SNR')

        # group the LC df by ('x1','color','z','band') and estimate sums

        if self.verbose:
            print('Estimating sumgrp')

        """
        df = self.lcdf.groupby(['x1', 'color', 'z', 'band']).apply(
            lambda x: self.sumgrp(x)).reset_index()
        """
        # make groups and estimate sigma_Color for combinations of SNR per band

        if self.verbose:
            print('estimating dfsigmaC')
        """
        dfsigmaC = df.groupby(['x1', 'color', 'z']).apply(
                lambda x: self.sigmaC(x)).reset_index()
        """
        self.lcdf = self.lcdf.round({'x1': 1, 'color': 1, 'z': 2})

        groups = self.lcdf.groupby(['x1', 'color', 'z'])

        if self.verbose:
            print('groups', len(groups))
            for name, grp in groups:
                print(name)

        dfsigmaCb = self.lcdf.groupby(['x1', 'color', 'z']).apply(
            lambda x: self.sigmaC_all(x)).reset_index()

        print('Done sigma', dfsigmaCb.columns)
        """
        cols = ['z']
        for val in ['SNRcalc','flux_5_e_sec']:
            for b in self.bands:
                cols.append('{}_{}'.format(val,b))
        """
        return dfsigmaCb

    def get_SNR(self, grp):
        """
        Method to get SNR for all the bands

        Parameters
        ---------------
        grp : pandas df group

        Returns
        ----------
        dict of bands and SNR values
        """
        # init SNR values to zero

        z = grp.name[2]
        if self.verbose:
            print('in get_SNR', z)

        SNR = {}
        for b in self.bands:
            SNR[b] = [0.0]

        # identify bands of interest and set SNRs
        # bands not present have SNR equal to 0.0
        # others should have a minimum SNR that would correspond to a single visit

        dictband = {}

        # SNR_min = 10.
        SNR_max = self.SNR_par['max']

        """
        if z >= 0.65:
            SNR_min = 20.

        SNR_min = 1.

        """

        # generate SNR values of interest (per band)
        for band in grp['band'].unique():
            idx = grp['band'] == band
            dictband[band] = grp[idx]
            # Get SNR_min
            idxb = self.medm5['filter'] == band
            m5_single = self.medm5[idxb]['fiveSigmaDepth'].values
            # print(band,'m5single', m5_single)
            SNR_min = self.SNR_from_m5[band]((m5_single, z)).item()
            """
            if SNR_min == [0.]:
                SNR_min = self.snrdictb[band][0].tolist()
                print('go there',SNR_min)
            """
            if SNR_min >= SNR_max or (SNR_max-SNR_min) < 1.:
                SNR[band] = np.array([SNR_min])
            else:
                SNR[band] = np.arange(np.round(SNR_min, 0),
                                      SNR_max, self.SNR_par['step'])

            # need to clean SNR here : if the corresponding number of visits is zero

            m5 = self.m5_from_SNR[band]((SNR[band], [z]*len(SNR[band])))

            nvisits = 10**((m5-m5_single)/1.25)
            # print(band, m5, type(m5),nvisits,m5_single)
            idx = m5 > 0.
            idx &= nvisits <= 110.
            # print(band, m5, type(m5),m5[idx],type(SNR[band]),SNR[band][idx])
            # print(band,z,SNR[band],type(SNR[band]))
            SNR[band] = SNR[band][idx].tolist()
            if band == 'g':
                SNR[band].append(100000.)
            # SNR[band] = list(np.arange(SNR_min, SNR_min+10, 10))
            # SNR[band] = [SNR_min]
            """
            if band == 'y':
                SNR[band] = [0.]
            """
            """
            if band == 'y':
                snrlist = list(np.arange(0., SNR_max, self.SNR_par['step']))
                snrlist[0] = 0.0001
                SNR[band] = snrlist
            """
            if z <= 0.3:
                SNR['z'] = [0.0]
            if z <= 0.5:
                SNR['y'] = [0.0]

        """
        for b in grp['band'].unique():
            SNR[b] = [20]
        """
        """
        SNR['g'] = [83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]
        SNR['r'] =  [83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]
        SNR['i'] =  [72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
            80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]
        SNR['z'] = [0.0]
        SNR['y'] = [0.0]
        """
        """
        SNR['g'] = [0.]
        SNR['r'] =  [0.]
        SNR['i'] =  [40.]
        SNR['z'] =  [50.]
        SNR['y'] = [20.]
        """
        if self.verbose:
            print('SNR values', SNR)
        # SNR = dict(zip('grizy', [[0.], [25.], [25.], [30.], [35.]]))

        return dictband, SNR

    def sigmaC_all(self, grp):
        """
        For a given group (grp), this method estimates sigmaC
        for a set of SNR-band combinations

        Parameters
        ----------
        grp: pandas df group

        Returns
        -------
        pandas df with the following columns:
        x1,color,z,band
        (sum_flux,SNR,self.listcol,flux_5_e_sec,m5_calc,flux_5)_band
        sigmaC,SNRcalc_tot


        """
        if self.verbose:
            print('Processing sigmaC', grp.name)

        x1 = grp.name[0]
        color = grp.name[1]
        z = grp.name[2]

        dictband, SNR = self.get_SNR(grp)

        SNR_split = self.splitSNR(SNR, nbands=5, nsplit=4)
        # SNR_split = self.splitSNR(SNR, nbands=-1, nsplit=3)
        if self.verbose:
            print('SNR_split', grp.name, SNR_split.keys())

        # for key, vals in SNR_split.items():
        #    print('SNR_split', key, vals)

        """
        dfres = pd.DataFrame()
        for key, vals in SNR_split.items():
            if self.verbose:
                print('Processing SNR',key,vals)
            resi = self.combiSNR(grp, dictband, vals, x1, color, z, key)
            if resi is not None:
                dfres = pd.concat((dfres, resi), ignore_index=True)
        """
        dfres = self.multiSNR(grp, SNR_split, dictband, x1, color, z)

        if self.verbose:
            print('final resu', dfres)

        # output

        cols = []
        for colname in ['SNRcalc', 'm5calc', 'fracSNR', 'flux_5_e_sec', 'Nvisits']:
            for band in 'grizy':
                cols.append('{}_{}'.format(colname, band))
        cols.append('Nvisits')

        if dfres.empty:
            mychan = pd.DataFrame([[-1.]*len(cols)], columns=cols)
            output = mychan.loc[0].reindex(cols)
            output = output.fillna(0.0)
        else:
            minPar = 'Nvisits'
            idx = int(dfres[[minPar]].idxmin())
            output = dfres.loc[idx].reindex(cols)
            output = output.fillna(0.0)

        return output

    def splitSNR(self, SNR, nbands=1, nsplit=3):
        """
        Method to split SNR dict

        Parameters
        ---------------
        SNR: dict
          dict of SNR values (key: band)
        nbands: int, opt
           number of SNR bands to split (default: 1)
        nsplit: int, opt
           number of splits per band

        Returns
        -----------
        dict: dict of splitted SNRs

        """

        SNR_band = {}

        if nbands == -1:
            SNR_band[0] = SNR
            return SNR_band

        for key in SNR.keys():
            SNR_band[key] = {}

        nb_split = 0
        for key, vals in SNR.items():
            if len(vals) >= nsplit and nb_split < nbands:
                rr = np.linspace(0, len(vals), nsplit+1, dtype='int')
                for io in range(len(rr)-1):
                    SNR_band[key][io] = vals[rr[io]: rr[io+1]]
                nb_split += 1
            else:
                SNR_band[key][0] = vals

        # now all combinations
        icombi = -1
        SNR_split = {}
        for ig in range(len(SNR_band['g'])):
            for ir in range(len(SNR_band['r'])):
                for ii in range(len(SNR_band['i'])):
                    for iz in range(len(SNR_band['z'])):
                        for iy in range(len(SNR_band['y'])):
                            icombi += 1
                            SNR_split[icombi] = {}
                            SNR_split[icombi]['g'] = SNR_band['g'][ig]
                            SNR_split[icombi]['r'] = SNR_band['r'][ir]
                            SNR_split[icombi]['i'] = SNR_band['i'][ii]
                            SNR_split[icombi]['z'] = SNR_band['z'][iz]
                            SNR_split[icombi]['y'] = SNR_band['y'][iy]

        return SNR_split

    def multiSNR(self, grp, SNR_split, dictband, x1, color, z):
        """
        Method to perform combination of band- SNRs using multiprocessing

        Parameters
        --------------


        Returns
        ----------



        """

        # creating outputdir if needed
        if self.save_SNR_combi:
            outdir_combi = '{}/z_{}'.format(self.dirSNR, z)
            if not os.path.isdir(outdir_combi):
                os.system('mkdir -p {}'.format(outdir_combi))

        # multiprocessing parameters
        ref = list(SNR_split.keys())
        nz = len(ref)
        t = np.linspace(0, nz, self.nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.loopSNR,
                                         args=(ref[t[j]:t[j+1]], grp, SNR_split, dictband, x1, color, z, j, result_queue))
                 for j in range(self.nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(self.nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = pd.DataFrame()

        # gather the results
        for key, vals in resultdict.items():
            restot = pd.concat((restot, vals), ignore_index=True)

        print('done here')
        return restot

    def loopSNR(self, keys, grp, SNR_split, dictband, x1, color, z, j=0, output_q=None):

        time_ref = time.time()
        dfres = pd.DataFrame()
        for key in keys:
            vals = SNR_split[key]
            if self.verbose:
                print('Processing SNR', key, vals)
            resi = self.combiSNR(grp, dictband, vals, x1, color, z, key)
            if resi is not None:
                if self.save_SNR_combi:
                    self.saveCombi(resi, x1, color, z, '{}_{}'.format(j, key))
                else:
                    dfres = pd.concat((dfres, resi), ignore_index=True)

        if not dfres.empty:
            minPar = 'Nvisits'
            dfres = dfres.sort_values(by=[minPar])
            no = np.min([len(dfres), 100])
            res = dfres[:no]
        else:
            res = dfres
        print('done processing', j, time.time()-time_ref)
        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def combiSNR(self, grp, dictband, SNR, x1, color, z, icombi):

        # existing bands
        bands = ''.join(list(dictband.keys()))

        # missing bands
        missing_bands = list(set(self.bands).difference(bands))

        # We are going to build a df with all SNR combinations
        # let us start with the first band
        df_ref = dictband[bands[0]]

        # make the SNR combination for this first band
        df_ref = self.addSNR_all(df_ref, SNR[bands[0]], bands[0], z)

        # print('there man',bands[0],  dictband[bands[0]],z,SNR[bands[0]],df_ref)
        # now make all the combinations
        df_tot = pd.DataFrame()
        df_merged = df_ref.copy()
        time_ref = time.time()

        for i in range(1, len(bands)):
            b = bands[i]
            df_to_merge = self.addSNR_all(dictband[b], SNR[b], b, z)
            if not df_tot.empty:
                df_merged = df_tot.copy()
            dfb = pd.DataFrame()
            for ikey in df_to_merge['key'].values:
                df_merged.loc[:, 'key'] = ikey
                dfb = pd.concat([dfb, df_merged], sort=False)

            df_tot = dfb.merge(df_to_merge, left_on=['key'], right_on=['key'])

        if self.verbose:
            print('after combi', time.time()-time_ref)

        listb = []
        for b in bands:
            listb.append('SNRcalc_{}'.format(b))

        df_tot['SNRcalc_tot'] = np.sqrt(
            (df_tot[listb]*df_tot[listb]).sum(axis=1))
        df_tot['bands'] = ''.join(bands)

        # time_ref = time.time()

        # Estimate sigma_Color for all the combinations build at the previous step
        for col in self.listcol:
            df_tot.loc[:, col] = df_tot.filter(
                regex='^{}'.format(col)).sum(axis=1)

        if self.verbose:
            vv = []
            for b in 'gri':
                for val in ['m5calc', 'SNRcalc']:
                    vv.append('{}_{}'.format(val, b))
            print('estimating sigmaC', df_tot[vv])
            """
            print(df_tot.columns)
            vv = []
            for b in 'grizy':
                for uu in ['SNRcalc','m5calc']:
                    vv.append('{}_{}'.format(uu,b))
            print(df_tot[vv])
            vv =  []
            for b in 'y':
                for bb in self.listcol:
                    vv.append('{}_{}'.format(bb,b))
            print(df_tot[vv])
            """

        df_tot['sigmaC'] = np.sqrt(CovColor(df_tot).Cov_colorcolor)

        if self.verbose:
            print('sigmaC', df_tot['sigmaC'])
        # add the missing bands to have a uniform format z-independent

        for b in missing_bands:
            for col in self.listcol:
                df_tot.loc[:, '{}_{}'.format(col, b)] = 0.0
            df_tot.loc[:, 'SNRcalc_{}'.format(b)] = 0.0
            df_tot.loc[:, 'flux_e_sec_{}'.format(b)] = 0.0
            df_tot.loc[:, 'm5calc_{}'.format(b)] = 0.0

        # select only combi with sigma_C ~ self.sigma_color_cut
        """
        idx = np.abs(df_tot['sigmaC'] -
                     self.sigma_color_cut) < 0.01*self.sigma_color_cut
        """
        dfres = self.SNRvisits(
            df_tot.copy(), missing_bands, x1, color, z, icombi)

        # select only combi with less than 200 visits per night

        idx = dfres['Nvisits'] <= 300
        df_tot = dfres[idx]

        # print('uuuu',df_tot[['Nvisits_r','Nvisits_i','Nvisits_z','Nvisits_y','sigmaC']])
        idx = df_tot['sigmaC'] >= 0.039
        idx = df_tot['sigmaC'] < 0.041

        if self.verbose:
            print('sigmaC_cut', len(df_tot[idx]))

        """
        if len(df_tot[idx]) < 1:
            # check whether you have enough SNR
            idx = df_tot['sigmaC'] < 0.04
            if len(df_tot[idx]) < 1:
                return None
        """
        if len(df_tot[idx]) < 1:
            return None

        # complete with the number of visits
        if self.verbose:
            print('completing the number of visits')

        """
        dfres = self.SNRvisits(
            df_tot[idx].copy(), missing_bands, x1, color, z, icombi)
        """
        if self.verbose:
            print('completing the number of visits', dfres)

        return df_tot[idx]

    def SNRvisits(self, dfres, missing_bands, x1, color, z, icombi):

        if self.verbose:
            print('in SNRvisits', self.bands)

        cols = []

        for b in self.bands:
            cols.append('Nvisits_{}'.format(b))
            # dfres.loc[:, 'fracSNR_{}'.format(b)] = dfres['SNRcalc_{}'.format(b)]/dfres['SNRcalc_tot']
            if self.medm5 is not None:
                m5single = self.medm5[self.medm5['filter']
                                      == b]['fiveSigmaDepth'].values
                dfres['m5single_{}'.format(b)] = m5single.tolist()*len(dfres)

            dfres.loc[:, 'Nvisits_{}'.format(
                b)] = 10**(0.8*(dfres['m5calc_{}'.format(b)]-dfres['m5single_{}'.format(b)]))

        if self.verbose:
            print('in SNR visits', dfres)

        for b in missing_bands:
            dfres.loc[:, 'Nvisits_{}'.format(b)] = 0.0

        dfres = dfres.fillna(0.0)
        # total number of visits
        dfres.loc[:, 'Nvisits'] = dfres[cols].sum(axis=1)

        if self.verbose:
            print('in SNR visits - total ', dfres)
        """
        # this is to dump results of combination
        if self.save_SNR_combi:
            self.saveCombi(dfres, x1, color, z, icombi)
        """
        return dfres

    def saveCombi(self, dfres, x1, color, z, icombi):
        """
        Method to save results of SNR combination in numpy file

        Parameters
        ---------------
        dfres: pandas df
           data to save

        """

        # grcp = dfres.copy()
        # grcp = grcp.sort_values(by=[minPar, 'Nvisits_y'])
        # grcp = grcp.fillna(value=0.)

        outdir_combi = '{}/z_{}'.format(self.dirSNR, z)
        nameOut = '{}/SNR_combi_{}_{}_{}_{}.npy'.format(outdir_combi,
                                                        x1, color, np.round(z, 2), icombi)

        if self.verbose:
            print('Saving', x1, color, z, icombi, len(dfres))
        np.save(nameOut, dfres.to_records(index=False))

    def sigmaC(self, grp):
        """
        For a given group (grp), this method estimates sigmaC
        for a set of SNR-band combinations

        Parameters
        ----------
        grp: pandas df group

        Returns
        -------
        pandas df with the following columns:
        x1,color,z,band
        (sum_flux,SNR,self.listcol,flux_5_e_sec,m5_calc,flux_5)_band
        sigmaC,SNRcalc_tot


        """
        if self.verbose:
            print('Processing sigmaC', grp.name)

        dictband, SNR = self.get_SNR(grp)

        # existing bands
        bands = ''.join(list(dictband.keys()))

        # missing bands
        missing_bands = list(set(self.bands).difference(bands))

        # We are going to build a df with all SNR combinations
        # let us start with the first band
        df_ref = dictband[bands[0]]

        # make the SNR combination for this first band
        df_ref = self.addSNR(df_ref, SNR[bands[0]], bands[0])

        # now make all the combinations
        df_tot = pd.DataFrame()
        df_merged = df_ref.copy()
        time_ref = time.time()

        for i in range(1, len(bands)):
            b = bands[i]
            df_to_merge = self.addSNR(dictband[b], SNR[b], b)
            if not df_tot.empty:
                df_merged = df_tot.copy()
            dfb = pd.DataFrame()
            for ikey in df_to_merge['key'].values:
                df_merged.loc[:, 'key'] = ikey
                dfb = pd.concat([dfb, df_merged], sort=False)

            df_tot = dfb.merge(df_to_merge, left_on=[
                'key'], right_on=['key'])

        if self.verbose:
            print('after combi', time.time()-time_ref)

        listb = []
        for b in bands:
            listb.append('SNRcalc_{}'.format(b))

        df_tot['SNRcalc_tot'] = np.sqrt(
            (df_tot[listb]*df_tot[listb]).sum(axis=1))
        df_tot['bands'] = ''.join(bands)

        # time_ref = time.time()

        # Estimate sigma_Color for all the combinations build at the previous step
        for col in self.listcol:
            df_tot.loc[:, col] = df_tot.filter(
                regex='^{}'.format(col)).sum(axis=1)

        df_tot['sigmaC'] = np.sqrt(CovColor(df_tot).Cov_colorcolor)

        # add the missing bands to have a uniform format z-independent
        for b in missing_bands:
            for col in self.listcol:
                df_tot.loc[:, '{}_{}'.format(col, b)] = 0.0
            df_tot.loc[:, 'SNRcalc_{}'.format(b)] = 0.0
            df_tot.loc[:, 'flux_e_sec_{}'.format(b)] = 0.0
            df_tot.loc[:, 'm5calc_{}'.format(b)] = 0.0

            # select only combi with sigma_C ~ self.sigma_color_cut
        idx = np.abs(df_tot['sigmaC'] -
                     self.sigma_color_cut) < 0.01*self.sigma_color_cut

        # that's it - return the results
        print('Done with', grp.name, time.time()-time_ref, df_tot.columns)
        return df_tot[idx]

    def addSNR_all(self, df, SNR, b, z):
        """
        Method add SNR-band combinations
        the five-sigma flux and corresponding m5 are estimated (SNR dependent).
        Fisher elements are also re-evaluated (f5 dependence through LC flux errors)

        Parameters
        ----------
        df: pandas df
         input data
        SNR: float
         SNR values
        b: str
         band

        Returns
        -------
        pandas df generated from the set of combinations

        """

        r = []
        df_tot = pd.DataFrame()

        for i, val in enumerate(SNR):
            r.append((i, val))
            df_cp = df.copy()
            df_cp.loc[:, 'key'] = i
            df_tot = pd.concat([df_tot, df_cp])

        df_SNR = pd.DataFrame(r, columns=['key', 'SNRcalc'])

        df_tot = df_tot.merge(df_SNR, left_on='key', right_on='key')

        # estimate m5 from SNR

        df_tot['m5calc'] = self.m5_from_SNR[b](
            (df_tot['SNRcalc'].values, [z]*len(df_tot)))

        # now need to estimate the phot error for this m5
        df_tot['SNR_indiv'] = 1. / \
            srand(self.gamma[b](df_tot['m5calc']),
                  df_tot['mag'], df_tot['m5calc'])

        # get the total SNR
        df_tot['SNR_indiv_tot'] = self.SNR_combi(
            df_tot['SNR_indiv'], df_tot['SNR_model'])
        # print('there man',b,len(df_tot),df_tot[['SNRcalc','m5calc','SNR_indiv_tot','SNR_model','SNR_indiv','flux_e_sec']])
        df_tot['fluxerr_indiv'] = df_tot['flux']/df_tot['SNR_indiv_tot']
        # update Fisher elements

        for col in self.listcol:
            df_tot[col] = df_tot[col.replace(
                'F', 'd')]/df_tot['fluxerr_indiv']**2
            """
            df_tot[col] = df_tot[col] * \
                (df_tot['SNR_indiv']/df_tot['snr_m5'])**2.
            """
        df_tot['SNRcalc'] = df_tot['SNRcalc'].round(decimals=1)
        df_tot['m5calc'] = df_tot['m5calc'].round(decimals=2)

        # grp = df_tot.groupby(['key', 'SNRcalc', 'm5calc'])[
        #    self.listcol+['flux_e_sec']].sum().reset_index()

        # print('hello',df_tot['SNR_indiv_tot'],self.SNR_min)
        idx = df_tot['SNR_indiv_tot'] >= 0.
        df_snr = df_tot[idx].copy()

        if len(df_snr) == 0:
            df_snr = pd.DataFrame(0, index=[0], columns=df_tot.columns)
        grp = df_snr.groupby(['key', 'band']).apply(
            lambda x: self.calc(x)).reset_index()

        # grp.loc[:, 'band'] = df_tot['band'].unique()
        # grp.loc[:, 'x1'] = df_tot['x1'].unique()
       # grp.loc[:, 'color'] = df_tot['color'].unique()

        # add suffix corresponding to the filter
        grp = grp.add_suffix('_{}'.format(b))
        # key here necessary for future merging
        grp = grp.rename(columns={'key_{}'.format(b): 'key'})

        return grp

    def SNR_combi(self, snra, snrb):
        """
        Method to estimate SNR from snra and snrb

        Parameters
        --------------
        snra: float
          first SNR
        snrb: float
          second SNR

        Returns
        ----------
        SNR such that 1/SNR**2=1/snra**2+1/snrb**2

        """

        snr = 1./(snra**2)+1./(snrb**2)

        return 1./np.sqrt(snr)

    def calc(self, grp):
        """
        Method to estimate few variables for the group grp

        Parameters
        --------------
        grp: pandas df

        Returns
        ----------
        pandas df with some estimated values

        """

        res = grp[self.listcol].sum()
        res['sumflux'] = np.sqrt(np.sum(grp['flux_e_sec']**2.))
        res['SNRcalc'] = grp['SNRcalc'].median()
        res['m5calc'] = grp['m5calc'].median()

        res['flux_5_e_sec'] = 0.
        if res['SNRcalc'] > 0.:
            res['flux_5_e_sec'] = 5.*res['sumflux']/res['SNRcalc']

        res = res.replace([np.inf, -np.inf], 0.0)

        return res

    def addSNR(self, df, SNR, b):
        """
        Method add SNR-band combinations
        the five-sigma flux and corresponding m5 are estimated (SNR dependent).
        Fisher elements are also re-evaluated (f5 dependence through LC flux errors)

        Parameters
        ----------
        df: pandas df
         input data
        SNR: float
         SNR values
        b: str
         band

        Returns
        -------
        pandas df generated from the set of combinations

        """

        r = []
        df_tot = pd.DataFrame()

        for i, val in enumerate(SNR):
            r.append((i, val))
            df_cp = df.copy()
            df_cp.loc[:, 'key'] = i
            df_tot = pd.concat([df_tot, df_cp])

        df_SNR = pd.DataFrame(r, columns=['key', 'SNRcalc'])

        df_tot = df_tot.merge(df_SNR, left_on='key', right_on='key')

        # df_tot = df_tot.drop(columns=['key'])

        # get the 5-sigma flux according to SNR vals
        df_tot['flux_5_e_sec'] = 5.*df_tot['sumflux']/df_tot['SNRcalc']

        # get m5 from the 5-sigma flux
        # this is for the background dominated case
        df_tot['m5calc'] = self.f5_to_m5[b](df_tot['flux_5_e_sec'])
        if b == 'z':
            ty = self.m5_from_SNR[b](
                (df_tot['SNRcalc'].values, df_tot['z'].values))

            print('hello', df_tot['m5calc'].values, ty, df_tot['sumflux'],
                  df_tot['SNRcalc'].values, df_tot['z'].values)
        # get the ratio of the 5-sigma fluxes: the one estimated two lines ago
        # and the original one, that is the one used for the original simulation
        # (and estimated from m5)

        df_tot['ratiof5'] = (df_tot['flux_5_e_sec']/df_tot['flux_5'])**2

        # correct Fisher matrix element from this ratio
        # so that these elements correspond to SNR values
        for col in self.listcol:
            df_tot[col] = df_tot[col]/df_tot['ratiof5']

        # drop the column used for the previous estimation
        df_tot = df_tot.drop(columns=['ratiof5'])

        # add suffix corresponding to the filter
        df_tot = df_tot.add_suffix('_{}'.format(b))
        # key here necessary for future merging
        df_tot = df_tot.rename(columns={'key_{}'.format(b): 'key'})

        # that is it - return the result

        return df_tot

    def plotSigmaC_SNR(self, dfsigmaC):
        """
        Plot sigmaC vs SNRtot vs z

        Parameters
        ----------
        dfsigmaC: pandas df
         data to plot

        Returns
        -------
        plot of sigmaC vs SNRtot (all SNR-band combis)
        one plot per z bin

        """

        for z in dfsigmaC['z'].unique():
            ii = dfsigmaC['z'] == z
            sel = dfsigmaC[ii]
            fig, ax = plt.subplots()
            fig.suptitle('z={}'.format(np.round(z, 2)))
            ax.plot(sel['SNRcalc_tot'], sel['sigmaC'], 'ko', mfc='None')
            # ax.plot(sel['Nvisits'],sel['sigmaC'],'ko',mfc='None')
            ax.set_xlabel(r'SNR$_{tot}$')
            ax.set_ylabel(r'$\sigma_{C}$')
            xlims = ax.get_xlim()
            ax.plot(xlims, [self.sigma_color_cut]*2, color='r')
            # ax.legend()

    def SNR(self, dfsigmaC):
        """
        Method to select SNR-band combis according to three reqs:

        - fracfluxband: SNR distrub similar to the flux fraction (per band)
        - nvisits_min: SNR combis that minimizes the total number of visits
        - nvisits_y_min: SNR combis that minimizes the total number of visits in the y band

        Parameters
        ---------------
        dfsigmaC: pandas df
         data to process

        Returns
        ----------
        pandas df with the SNR-band combi corresponding to the 3 above mentioned conditions

        """

        if self.verbose:
            print('SNR_z - SNR method')

        # select only combi with sigma_C ~ 0.04
        idx = np.abs(dfsigmaC['sigmaC'] -
                     self.sigma_color_cut) < 0.01*self.sigma_color_cut
        # idx = np.abs(dfsigmaC['sigmaC']-0.04)<0.1

        sel = dfsigmaC[idx].copy()

        # for these combi : estimate the SNR frac per band

        for b in self.bands:
            sel.loc[:, 'fracSNR_{}'.format(
                b)] = sel['SNRcalc_{}'.format(b)]/sel['SNRcalc_tot']
            if self.medm5 is not None:
                sel.loc[:, 'm5single_{}'.format(
                    b)] = self.medm5[self.medm5['filter'] == b]['fiveSigmaDepth'].values

        """
        if self.SNR_par['choice'] == 'fracflux':
            res = sel.groupby(['x1','color','z']).apply(
                lambda x: self.SNR_redshift(x)).reset_index()
        else:
            res = sel.groupby(['x1','color','z']).apply(
                lambda x: self.SNR_visits(x)).reset_index()
        """

        res = {}
        """
        if self.verbose:
            print('SNR_z-SNR -> SNR choice: fracflux')
        res['fracflux'] = sel.groupby(['x1', 'color', 'z']).apply(
            lambda x: self.SNR_redshift(x)).reset_index()
        """
        # for vv in ['Nvisits', 'Nvisits_y']:
        for vv in ['Nvisits']:
            if self.verbose:
                print('SNR_z-SNR -> SNR choice: {}'.format(vv))
            res[vv] = sel.groupby(['x1', 'color', 'z']).apply(
                lambda x: self.SNR_visits(x, vv)).reset_index()

        return res

    def SNR_visits(self, grp, minPar='Nvisits'):
        """
        Method to estimate the SNR-band combination that minimizes minPar

        Parameters
        ----------
        grp: pandas df group
         data to process
        minPar: str, opt
         parameter used to minimize (default: Nvisits)

        Returns
        -------
        pandas df with the SNR combination corresponding to the above-mentioned criteria

        """

        # Estimate the requested number of visits (per band and in total)
        cols = []
        for b in self.bands:
            cols.append('Nvisits_{}'.format(b))

        for b in self.bands:
            grp.loc[:, 'Nvisits_{}'.format(
                b)] = 10**(0.8*(grp['m5calc_{}'.format(b)]-grp['m5single_{}'.format(b)]))
            grp['Nvisits_{}'.format(
                b)] = grp['Nvisits_{}'.format(b)].fillna(0.0)

        grp.loc[:, 'Nvisits'] = grp[cols].sum(axis=1)

        print('nvisits', grp['Nvisits'])
        idx = int(grp[[minPar]].idxmin())

        if self.save_SNR_combi:

            grcp = grp.copy()
            grcp = grcp.sort_values(by=[minPar, 'Nvisits_y'])
            grcp = grcp.fillna(value=0.)

            # print('sorted', grcp, grcp[:1], grp.name)
            nameOut = '{}/SNR_combi_{}_{}_{}.npy'.format(self.dirSNR,
                                                         grp.name[0], grp.name[1], np.round(grp.name[2], 2))
            print('SAVING DATA')
            np.save(nameOut, grcp.to_records(index=False))
            # tab['X'] = (tab['Nvisits']/50.)**2+(tab['Nvisits_y']/tab['Nvisits'])**2 + \
            #       (tab['Nvisits_i']/tab['Nvisits_z'])**2

            # tab = tab.sort_values(by='X')
        # idxa = int(grp[['Nvisits']].idxmin())
        # idxb = int(grp[['Nvisits_y']].idxmin())

        cols = []
        for colname in ['SNRcalc', 'm5calc', 'fracSNR', 'flux_5_e_sec']:
            for band in 'grizy':
                cols.append('{}_{}'.format(colname, band))

        output = grp.loc[idx].reindex(cols)
        output = output.fillna(0.0)
        # colindex = [grp.columns.get_loc(c) for c in cols if c in grp]
        # output = grcp[:-1][cols]
        # print('there man', idx, cols, output, grcp[:1][cols])
        return output
        # return grp.loc[idxa,cols], grp.loc[idxb,cols]

    def SNR_redshift(self, grp):
        """
        Method to estimate the SNR-band combination
        with a distribution closest to flux fraction per band

        Parameters
        ----------
        grp: pandas df group

        Returns
        -------
        pandas df with the SNR combination corresponding to the above-mentioned criteria

        """

        ik = np.abs(self.fracSignalBand['z']-grp.name[2]) < 1.e-5
        selband = self.fracSignalBand[ik]

        r = []
        for b in self.bands:
            iop = selband['band'] == b
            if len(selband[iop]) == 0:
                r.append(0.0)
            else:
                r.append(selband[iop]['fracfluxband'].values[0])

        # print(r,grp[['fracSNR_i','fracSNR_z']].shape)

        cldist = []
        for b in self.bands:
            cldist.append('fracSNR_{}'.format(b))

        closest_index = distance.cdist([r], grp[cldist])  # .argmin()

        cols = []
        for colname in ['SNRcalc', 'm5calc', 'fracSNR', 'flux_5_e_sec']:
            for band in self.bands:
                cols.append('{}_{}'.format(colname, band))

        colindex = [grp.columns.get_loc(c) for c in cols if c in grp]

        return grp.iloc[closest_index.argmin(), colindex]


class Nvisits_z_plot:
    """
    class to display Number of visits vs redshift

    Parameters
    ---------------
    filename: str
      name of the file with data

    """

    def __init__(self, filename):

        self.data = pd.DataFrame(np.load(filename, allow_pickle=True))

        self.plot()

    def plot(self):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ls = dict(zip(range(1, 5), ['solid', 'dotted', 'dashed', 'dashdot']))
        color = dict(zip(['visits', 'chisq'], ['r', 'b']))
        for cadence in np.unique(self.data['cadence']):
            idx = np.abs(self.data['cadence']-cadence) < 1.e-8
            sela = self.data[idx]
            for min_par in np.unique(sela['min_par']):
                idxa = sela['min_par'] == min_par
                selb = sela[idxa]
                res = selb.groupby(['z']).apply(lambda x: pd.DataFrame(
                    {'Nvisits': [x['Nvisits'].sum()]})).reset_index()
                print(res)
                ax.plot(res['z'], res['Nvisits'],
                        ls=ls[int(cadence)], color=color[min_par])

        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format('Nvisits'))
        ax.set_xlim([0.3, 0.85])
        ax.set_ylim([0., 150.])
        plt.show()


class SNR_plot_deprecated:

    def __init__(self, SNRDir, x1, color,
                 SNR_step,
                 cutoff,
                 cadence,
                 theDir,
                 m5_file,
                 m5_type):
        """
        class to plot SNR results

        Parameters
        ----------
        SNRDir: str
         location directory of the SNR files
        x1: float
         SN stretch parameter
        color: float
         SN color parameter
        SNR_step: float
         SNR step used when scanning the SNR-band parameter space
        blue_cutoff: float
         wavelength cutoff(blue) applied to the data used for SNR-b estimation
        red_cutoff: float
         wavelength cutoff(red) applied to the data used for SNR-b estimation
        cadence: float
         cadence choice for the display of the results

        """

        # load parameters
        self.SNRDir = SNRDir
        self.x1 = x1
        self.color = color
        self.SNR_step = SNR_step
        self.cutoff = cutoff
        self.cadence = cadence
        self.theDir = theDir
        self.m5_file = m5_file
        self.m5_type = m5_type

        # load SNR files
        self.dictplot = self.load()

    def nameFile(self, SNRDir,
                 x1, color,
                 SNR_step,
                 cutoff,
                 bands,
                 SNR_choice):
        """
        Method defining the name of SNR file

        Parameters
        ----------
        SNRDir: str
         location directory of the SNR files
        x1: float
         SN stretch parameter
        color: float
         SN color parameter
        SNR_step: float
         SNR step used when scanning the SNR-band parameter space
        blue_cutoff: float
         wavelength cutoff(blue) applied to the data used for SNR-b estimation
        red_cutoff: float
         wavelength cutoff(red) applied to the data used for SNR-b estimation
        bands: str
         filters considered for SNR-b estimation
        SNR_choice: str
         type of choice used to get the SNR-b combination
        theDir: str
         location directory of m5 files
        m5_file: str
         m5 file name
        m5_type: str
         type of m5 values used

        Returns
        ------
        str: name of the SNR file

        """

        name = '{}/SNR_{}_{}_{}_{}_{}_{}.npy'.format(SNRDir,
                                                     x1, color,
                                                     SNR_step,
                                                     cutoff,
                                                     bands,
                                                     SNR_choice)

        return name

    def load(self):
        """
        Method to load SNR files

        Parameters
        ----------

        Returns
        -------
        dictplot: dict
         keys: SNR_choice_bands
         vals: number of visits(per band and in total) vs z

        """

        dictplot = {}
        """
       for SNR_choice in [('fracflux', 'rizy'),
                           ('Nvisits', 'rizy'),
                           ('Nvisits_y', 'rizy')]:
        """
        # ('Nvisits','riz')]:
        for SNR_choice in [('Nvisits', 'grizy')]:
            SNRNameb = self.nameFile(self.SNRDir,
                                     self.x1, self.color,
                                     self.SNR_step,
                                     self.cutoff,
                                     SNR_choice[1],
                                     SNR_choice[0])

            print('loading', SNRNameb)
            SNRb = pd.DataFrame(np.load(SNRNameb, allow_pickle=True))

            # myvisits = Nvisits_m5(SNRb,medValues)
            # m5_type = 'median_m5_filter'

            myvisits = Nvisits_cadence(
                SNRb, self.cadence, self.theDir, self.m5_file, self.m5_type, SNR_choice[0], SNR_choice[1]).nvisits_cadence
            print('visits', myvisits)
            dictplot['_'.join([SNR_choice[0], SNR_choice[1]])] = myvisits
        return dictplot

    def plotSummary(self):
        """
        Summary plot of the SNR results

        Parameters
        ----------

        Returns
        ------
        plot of the number of visits(total) vs z for the various configuration(SNR_choices)

        """

        ls = ['-', '--', '-.', ':']
        colors = ['k', 'b', 'r', 'g']

        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days'.format(self.cadence))

        keys = list(self.dictplot.keys())
        for key, data in self.dictplot.items():
            sel = data

            ax.plot(sel['z'].values, sel['Nvisits'].values,
                    color=colors[keys.index(key)], ls=ls[keys.index(key)],
                    label='{}'.format(key))
        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format('Nvisits'))
        ax.set_xlim([0.3, 0.85])
        rplot = []
        rplot.append((40, '6% - 10 seasons * 6 fields', -17))
        rplot.append((48, '6% - 10 seasons * 5 fields', +15))
        rplot.append((199, '6% - 2 seasons * 6 fields', -17))
        rplot.append((239, '6% - 2 seasons * 5 fields', +15))

        for val in rplot:
            ax.plot(ax.get_xlim(), [val[0]]*2, color='m')
            ax.text(0.33, val[0]+val[2], val[1])

    def plotSummary_band(self, bands='rizy', legy='Nvisits'):
        """
        Summary plot of the SNR results

        Parameters
        ----------

        Returns
        ------
        plot of the number of visits(per band) vs z for the various configuration(SNR_choices)

        """

        ls = ['-', '--', '-.', ':']
        colors = ['k', 'b', 'r', 'g']

        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days'.format(self.cadence))
        keys = list(self.dictplot.keys())

        io = -1
        for key, data in self.dictplot.items():
            io += 1
            cols = []
            for b in key.split('_')[-1]:
                cols.append('Nvisits_{}'.format(b))
            sel = data

            for b in key.split('_')[-1]:
                if key == keys[0]:
                    ax.plot(sel['z'].values, sel['Nvisits_{}'.format(b)].values,
                            color=filtercolors[b],
                            label=b, ls=ls[keys.index(key)])
                else:
                    ax.plot(sel['z'].values, sel['Nvisits_{}'.format(b)].values,
                            color=filtercolors[b],
                            ls=ls[keys.index(key)])
            yval = 150.-10*io
            ax.plot([0.3, 0.35], [yval]*2, ls[keys.index(key)], color='k')
            ax.text(0.37, yval, key)

        ax.legend()
        ax.grid()
        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))

    def plotIndiv(self, config, bands='rizy', legy='N$_{visits}$/field/observing night'):
        """
        plot individual(per SNR_choice) results

        Parameters
        ----------
        config: str
         config(=SNR_choice_band) chosen
        bands: str
         filters to plot
        legy: str, opt
         ylabel(default: 'N$_{visits}$/field/observing night')
         could also be: 'Filter allocation'

        Returns
        ------
        plot of the number of visits the number of visits or the filter allocation.

        """
        fig, ax = plt.subplots()
        fig.suptitle('cadence = {} days - {}'.format(self.cadence, config))

        keys = list(self.dictplot.keys())
        print(keys)
        data = self.dictplot[config]

        cols = []
        for b in config.split('_')[-1]:
            cols.append('Nvisits_{}'.format(b))

        for b in bands:
            if legy == 'Filter allocation':
                ax.plot(data['z'], data['Nvisits_{}'.format(
                    b)]/data['Nvisits'], color=filtercolors[b], label='{}'.format(b))
            else:
                ax.plot(data['z'], data['Nvisits_{}'.format(b)],
                        color=filtercolors[b], label='{}'.format(b))

        ax.set_xlabel(r'z')
        ax.set_ylabel(r'{}'.format(legy))
        ax.legend()
        ax.grid()
