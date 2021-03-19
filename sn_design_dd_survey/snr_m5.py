import os
import pandas as pd
import numpy as np
from .utils import m5_to_flux5, srand, gamma, load
import multiprocessing
from scipy.interpolate import interp1d


class SNR_m5:
    """
    Class to estimate, for each band and each considered LC
    the Signal-to-Noise Ratio vs fiveSigmaDepth

    Parameters
    ---------------
    inputDir: str
      input directory where the LC file is located
    refFile: str
      name of the LC file
    outfile: str, opt
      output file name (default: SNR_m5)
    snrmin: float, opt
      min snr for SNR LC estimation

    """

    def __init__(self, inputDir, refFile, outfile='SNR_m5.py', snrmin=1.):

        self.outfile = outfile
        self.snrmin = snrmin

        self.process_main(inputDir, refFile)

        """
        resdf = pd.DataFrame(np.copy(np.load(outfile, allow_pickle=True)))
        self.get_m5(resdf)
        self.plot(resdf)
        """

    def process_main(self, inputDir, refFile):

        # load the reference file

        refdata = pd.DataFrame(np.copy(load(inputDir, refFile)))
        refdata['band'] = refdata['band'].map(lambda x: x.decode()[-1])
        print('loading', inputDir, refFile, refdata['z'].max())
        self.error_model = 0.
        if 'error_model' in refFile:
            self.error_model = 1

        """
        idc = (refdata['x1']-x1) < 1.e-5
        idc &= (refdata['color']-color) < 1.e-5
        refdata = refdata[idc]
        """

        # load the gamma file
        #gamma = self.load('reference_files', 'gamma.hdf5')

        # load mag to flux corresp

        # mag_to_flux = np.load('reference_files/Mag_to_Flux_SNCosmo.npy')
        mag_to_flux = m5_to_flux5('grizy')

        # print(mag_to_flux.dtype)

        # select exposure time of 30s and
        #idx = np.abs(gamma['exptime']-30.) < 1.e-5
        #selgamma = gamma[idx]

        bands = 'grizy'

        # get interpolators for gamma and magflux
        gammadict = gamma(bands)
        # magfluxdict = {}
        """
        for b in bands:
            io = selgamma['band'] == b
            gammadict[b] = interp1d(
                selgamma[io]['mag'], selgamma[io]['gamma'], bounds_error=False, fill_value=0.)
           
        """
        # SNR vs m5 estimation

        zref = 0.8
        result_queue = multiprocessing.Queue()
        for j, b in enumerate(bands):
            idx = refdata['band'] == b
            #idx &= np.abs(refdata['z']-zref) < 1.e-5
            datab = refdata[idx]
            p = multiprocessing.Process(name='Subprocess-{}'.format(b), target=self.process,
                                        args=(datab, gammadict[b], mag_to_flux[b], j, result_queue))
            p.start()
            #res = self.process(datab, gammadict[b], mag_to_flux[b])
            #resdf = pd.concat((resdf, res))

        resultdict = {}
        # get the results in a dict

        for i in range(len(bands)):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        resdf = pd.DataFrame()
        # gather the results
        for key, vals in resultdict.items():
            resdf = pd.concat((resdf, vals), sort=False)

        # save the result in a numpy array
        np.save(self.outfile, resdf.to_records(index=False))

    def process(self, data, gamma, magtoflux, j=0, output_q=None):
        """
        Method to estimate SNR vs m5

        Parameters
        ---------------
        data: pandas df
          LC DataFrame
        gamma: interp1d
           gamma values interpolator (m5->gamma)
        magtoflux: interp1d
           mag to flux interpolator (mag -> flux)

        Returns
        ----------
        pandas df with the following cols:
        band, z, SNR, SNR_bd, m5 where
        - SNR = 1/srand
        - SNR_bd = SNR if background dominated (sigma = sigma_5 = flux_5/5.
        where flux_5 is estimated from m5)

        """
        res = pd.DataFrame()
        datab = data.copy()

        idc = datab['flux'] > 1.e-10

        if self.error_model:
            idc &= datab['fluxerr_model'] >= 0.
        datab = datab[idc]

        # set the error model error to the one corresponding to the flux in elec/sec
        datab['SNR_model'] = datab['flux']/datab['fluxerr_model']

        for m5 in np.arange(15., 28., 0.01):
            # for m5 in [24]:
            datab.loc[:, 'm5'] = m5
            datab.loc[:, 'gamma'] = gamma(m5)
            datab.loc[:, 'SNR_photo'] = 1. / \
                srand(datab['gamma'],
                      datab['mag'], datab['m5'])

            datab.loc[:, 'SNR_photo_bd'] = (
                5.*datab['flux_e_sec'])/magtoflux(m5)

            datab.loc[:, 'SNR'] = self.SNR(
                datab['SNR_model'], datab['SNR_photo'])

            datab.loc[:, 'SNR_bd'] = self.SNR(
                datab['SNR_model'], datab['SNR_photo_bd'])

            datab.loc[:, 'f5'] = magtoflux(m5)

            # print(datab[['SNR','SNR_bd','SNR_model','SNR_photo','SNR_photo_bd']])
            """
            import matplotlib.pyplot as plt
            plt.plot(datab['phase'],datab['SNR'],'ko')
            plt.show()
            """
            # select only LC points with SNR>=snrmin
            idsnr = datab['SNR_photo_bd'] >= self.snrmin
            grp = datab[idsnr].groupby(['band', 'z']).apply(
                lambda x: self.calc(x, m5)).reset_index()
            res = pd.concat((res, grp))

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def SNR(self, snra, snrb):
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

    def calc(self, grp, m5):
        """
        Method to estimate some quantities per group (1 group = 1LC/z/band)

        Parameters
        --------------
        grp: pandas grp
          data to process
        m5: float
          fiveSigmaDepth value

        Returns
        -----------
        pandas df with the following cols:
         - SNR = np.sqrt(np.sum(SNR*SNR)
        - SNR_bd = np.sqrt(np.sum(SNR_bd*SNR_bd)
        - m5: fiveSigmaDepth

        """
        sumflux = np.sqrt(np.sum(grp['flux_e_sec']**2.))
        SNR = np.sqrt(np.sum(grp['SNR']**2))
        SNR_bd = np.sqrt(np.sum(grp['SNR_bd']**2))
        SNR_photo = np.sqrt(np.sum(grp['SNR_photo']**2))
        SNR_photo_bd = np.sqrt(np.sum(grp['SNR_photo_bd']**2))
        SNR_test = 5.*sumflux/grp['f5'].median()

        return pd.DataFrame({'SNR': [SNR],
                             'SNR_bd': [SNR_bd],
                             'SNR_test': [SNR_test],
                             'SNR_photo': [SNR_photo],
                             'SNR_photo_bd': [SNR_photo_bd],
                             'm5': [m5]})

    def plot(self, data, zref=0.7):
        """
        Method to plot SNR vs m5

        """

        import matplotlib.pyplot as plt
        fontsize = 15
        for b in 'grizy':
            idx = data['band'] == b
            sel = data[idx]
            fig, ax = plt.subplots(nrows=2)
            fig.suptitle('{} band - z = {}'.format(b, zref), fontsize=fontsize)
            idxb = np.abs(sel['z']-zref) < 1.e-5
            selb = sel[idxb]
            ax[0].plot(selb['m5'], selb['SNR'],  color='k',
                       label='SNR')
            ax[0].plot(selb['m5'], selb['SNR_bd'], color='r',
                       label='background dominated')
            ax[0].plot(selb['m5'], selb['SNR_test'], color='b',
                       label='background dominated - test')

            # ax[0].set_xlabel('m$_{5}$ [mag]', fontsize=fontsize)
            ax[0].set_ylabel('SNR', fontsize=fontsize)
            ax[0].legend(fontsize=fontsize)
            ax[0].yaxis.set_tick_params(labelsize=15)
            # ax[0].xaxis.set_tick_params(labelsize=15)

            ax[1].plot(selb['m5'], selb['SNR']/selb['SNR_bd'],  color='k',
                       label='SNR/SNR_bd')

            ax[1].set_xlabel('m$_{5}$ [mag]', fontsize=fontsize)
            ax[1].set_ylabel('SNR ratio', fontsize=fontsize)
            ax[1].legend(fontsize=fontsize)
            ax[1].yaxis.set_tick_params(labelsize=15)
            ax[1].xaxis.set_tick_params(labelsize=15)

        plt.show()

    def get_m5(self, data, SNR=dict(zip('grizy', [20., 20., 30., 30., 35.])), zref=0.7):

        for b in SNR.keys():
            idx = data['band'] == b
            idx &= np.abs(data['z']-zref) < 1.e-5
            sel = data[idx]
            if len(sel) > 0:
                myinterpa = interp1d(
                    sel['SNR'], sel['m5'], bounds_error=False, fill_value=0.)
                myinterpb = interp1d(
                    sel['SNR_bd'], sel['m5'], bounds_error=False, fill_value=0.)
                myinterpc = interp1d(
                    sel['SNR_test'], sel['m5'], bounds_error=False, fill_value=0.)
                print(b, SNR[b], myinterpa(SNR[b]),
                      myinterpb(SNR[b]), myinterpc(SNR[b]))
