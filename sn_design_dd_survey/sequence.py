from sn_design_dd_survey.wrapper import Data, Nvisits_cadence
from sn_design_dd_survey.snr import SNR, Nvisits_z_plot
from sn_design_dd_survey.signal_bands import RestFrameBands
from sn_design_dd_survey.templateLC import templateLC
from sn_design_dd_survey.snr_m5 import SNR_m5
from sn_design_dd_survey.ana_combi import CombiChoice, Visits_Cadence
from sn_design_dd_survey.zlim_visits import RedshiftLimit
from sn_tools.sn_io import loopStack
import numpy as np
import glob
import pandas as pd
import time
import multiprocessing
from astropy.table import Table, vstack


def cut_off(error_model, bluecutoff, redcutoff):

    cuto = '{}_{}'.format(bluecutoff, redcutoff)
    if error_model:
        cuto = 'error_model'

    return cuto


class TemplateData:
    """
    class to generate usefull data: template LC, snr_m5, ...

    Parameters
    ---------------
    x1: float, opt
      SN x1 (default=-2.0)
    color: float, opt
       SN color (default: 0.2)
    bands: str, opt
      bands to consider (default: grizy)
    dirStudy: str, opt
      main directory where files will be produced (default: dd_design)
    dirTemplates: str, opt
      sub dir where template LC will be placed
    dirSNR_m5: str, opt
      sub dir where SNR<->m5 files will be placed

    """

    def __init__(self, x1=-2., color=0.2,
                 bands='grizy',
                 dirStudy='dd_design',
                 dirTemplates='Templates',
                 dirSNR_m5='SNR_m5',):

        self.x1 = x1
        self.color = color
        self.bands = bands

        self.dirTemplates = '{}/{}'.format(dirStudy, dirTemplates)
        self.dirSNR_m5 = '{}/{}'.format(dirStudy, dirSNR_m5)

    def templates(self,
                  zmin=0,
                  zmax=1.1,
                  zstep=0.01,
                  error_model=1,
                  bluecutoff=380.,
                  redcutoff=800.,
                  ebvofMW=0.,
                  simulator='sn_fast',
                  cadence=3.):
        """
        Method to generate templates with a defined regular cadence

        Parameters
        --------------
        zmin :  float,opt
          min redshift value (default: 0)
        zmax: float,opt
          max redshift value (default: 1.1)
        zstep: float, opt
          step redshift value (default:0.01)
        error_model: int, opt
           error model for the simulation (default:1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        cadences: float, opt
          cadence of observation (the same filter for each filter) (default: 3.)
        """

        cutof = cut_off(error_model, bluecutoff, redcutoff)

        templid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(
            simulator, self.x1, self.color, ebvofMW, cutof, int(cadence))
        fname = 'LC_{}_0.hdf5'.format(templid)
        cadences = dict(zip(self.bands, [cadence]*len(self.bands)))
        # generate template here - no error model cut
        templateLC(self.x1, self.color, simulator, ebvofMW, bluecutoff, redcutoff,
                   error_model, -1.0,
                   zmin, zmax, zstep, self.dirTemplates, self.bands, cadences, templid)

    def snr_m5(self, snrmin=1., error_model=1, error_model_cut=-1.0, bluecutoff=380., redcutoff=800.):
        """
        Method to produce SNR vs m5 files

        Parameters
        --------------
        snrmin: float, opt
          SNR min selection
        error_model: int
          to activate or not the error moedl (default: 1)
        error_model_cut: float
         max error model flux (relative)(default: 0.1)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        """
        cutoff = cut_off(error_model, bluecutoff, redcutoff)
        search_path = '{}/LC*{}*_{}.hdf5'.format(
            self.dirTemplates, cutoff, np.round(error_model_cut, 2))
        template_list = glob.glob(search_path)

        for lc in template_list:
            lcName = lc.split('/')[-1]
            outName = '{}/{}'.format(self.dirSNR_m5,
                                     lcName.split('.hdf5')[0].replace('LC', 'SNR_m5'))
            print('processing', lcName)
            SNR_m5(self.dirTemplates, lcName, '{}.npy'.format(outName), snrmin)


class DD_SNR:

    def __init__(self, x1=-2., color=0.2, bands='grizy',
                 dirStudy='dd_design',
                 dirTemplates='Templates',
                 dirSNR_m5='SNR_m5',
                 dirm5='m5_files',
                 dirSNR_combi='SNR_combi',
                 dirSNR_opti='SNR_opti',
                 cadence=3,
                 error_model=1,
                 error_model_cut=-1.0,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 m5_file='medValues_flexddf_v1.4_10yrs_DD.npy'):

        self.x1 = x1
        self.color = color
        self.bands = bands

        self.dirTemplates = '{}/{}'.format(dirStudy, dirTemplates)
        self.dirSNR_m5 = '{}/{}'.format(dirStudy, dirSNR_m5)
        self.dirm5 = '{}/{}'.format(dirStudy, dirm5)
        self.dirSNR_combi = '{}/{}'.format(dirStudy, dirSNR_combi)
        self.dirSNR_opti = '{}/{}'.format(dirStudy, dirSNR_opti)

        # get data
        self.data = self.grab_data(cadence,
                                   error_model,
                                   error_model_cut,
                                   bluecutoff, redcutoff,
                                   ebvofMW,
                                   sn_simulator,
                                   m5_file)
        self.fracSignalBand = self.data.fracSignalBand.fracSignalBand

    def grab_data(self, cadence,
                  error_model,
                  error_model_cut,
                  bluecutoff, redcutoff,
                  ebvofMW,
                  sn_simulator,
                  m5_file):
        """
        Method to grab Data corresponding to a given cadence

        Parameters
        --------------
        cadence: int, opt
          cadence of the data to get (
        error_model: int, opt
           error model for the simulation (default:1)
        error_model_cut: float, opt
          selection error model cut with which templates have been generated (default: -1.)
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)
        ebvofMW: float, opt
          ebv of MW (default:0)
        simulator: str, opt
          simulator to use to produce LCs (default: 'sn_fast')
        m5_file: str,opt
          m5 file (default: 'medValues_flexddf_v1.4_10yrs_DD.npy')

        """

        self.cutoff = cut_off(error_model, bluecutoff, redcutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0_{}.hdf5'.format(
            sn_simulator, self.x1, self.color, ebvofMW, self.cutoff, int(cadence), np.round(error_model_cut, 2))
        m5Name = '{}/{}'.format(self.dirm5, m5_file)
        return Data(self.dirTemplates, lcName, m5Name, self.x1, self.color, bluecutoff, redcutoff, error_model, bands=self.bands)

    def plot_data(self, bluecutoff=380., redcutoff=800.):
        """
        Method to display useful plots related to data

        Parameters
        --------------
        bluecutoff: float, opt
          blue cutoff if error_model=0 (default:380)
        redcutoff: float, opt
          red cutoff if error_model=0 (default: 800.)

        """
        import matplotlib.pyplot as plt

        self.data.plotzlim()
        self.data.plotFracFlux()
        self.data.plot_medm5()

        # this is to plot restframebands cutoff
        mybands = RestFrameBands(blue_cutoff=bluecutoff,
                                 red_cutoff=redcutoff)
        mybands.plot()
        plt.show()

    def SNR_combi(self,
                  SNR_par=dict(
                      zip(['max', 'step', 'choice'], [70., 1., 'Nvisits'])),
                  zmin=0.1,
                  zmax=1.1,
                  zstep=0.05,
                  nproc=8):
        """
        Method to estimate SNR combinations

        Parameters
        --------------
        SNR_par: dict, opt
          parameters for SNR combi estimation
        zmin: float, opt
          min redshift (default: 0.1)
        zmax: float, opt
          max redshift (default: 1.1)
        zstep: float, opt
         step for redshift (default: 0.05)
        nproc: int, opt
          number of procs for multiprocessing

        """

        zref = np.round(np.arange(zmin, zmax+zstep, zstep), 2)

        SNR_name = (self.data.lcName
                    .replace('LC', 'SNR_m5')
                    .replace('.hdf5', '.npy')
                    )
        SNR_m5_file = '{}/{}'.format(self.dirSNR_m5, SNR_name)

        snr_calc = SNR(self.dirSNR_combi, self.data, SNR_par,
                       SNR_m5_file=SNR_m5_file, zref=zref,
                       save_SNR_combi=True, verbose=False, nproc=nproc)


def OptiCombi(fracSignalBand, dirStudy='dd_design',
              dirSNR_combi='SNR_combi',
              dirSNR_opti='SNR_opti',
              snr_opti_file='opti_combi.npy',
              nproc=8):
    """
    Function to select optimal (wrt a certain criteria) SNR combinations

    Parameters
    ---------------
    fracSignalBands: numpy array
      fraction of signal per band
    dirStudy: str, opt
      main dir for the study (default: dd_design)
    dirSNR_combi: str, opt
       SNR combi dir (default: SNR_combi)
    dirSNR_opti: str, opt
      location dir of the opti combi output file
    snr_opti_file: str, opt
      name of the output file containing optimal combinations
    nproc: int, opt
      number of proc to use for multiprocessing (default: 8)
    """
    dirSNR_combi_full = '{}/{}'.format(dirStudy, dirSNR_combi)
    combi = CombiChoice(fracSignalBand, dirSNR_combi_full)

    resdf = pd.DataFrame()
    snr_dirs = glob.glob('{}/*'.format(dirSNR_combi_full))

    for fi in snr_dirs:
        z = (
            fi.split('/')[-1]
            .split('_')[-1]
        )
        z = np.round(float(z), 2)

        res = combi(z, nproc)
        if res is not None:
            resdf = pd.concat((resdf, res))

    np.save('{}/{}/{}'.format(dirStudy, dirSNR_opti, snr_opti_file),
            resdf.to_records(index=False))


def Nvisits_Cadence_z(m5_single, snr_opti_file,
                      dirStudy='dd_design',
                      dirSNR_m5='SNR_m5',
                      dirSNR_opti='SNR_opti',
                      dirNvisits='Nvisits_z',
                      outName='Nvisits_z_med.npy'):
    """
    function to estimate the number of visits
    as a function of the cadence using the 'optimal' result.
    The idea is that the SNR optimized for a given cadence
    is the same for other cadences.

    Three ingredients needed:
    - m5 single visit (file used to make SNR combis)
    - SNR_m5 vs cadence
    - SNR_opti file

    Parameters
    ---------------
    m5_single: numpy array
      m5 single band (median per filter over seasons)
    snr_opti: str
      file name corresponding to SNR opti
    dirStudy: str, opt
      main dir for the study (default: dd_design)
    dirSNR_m5: str, opt
       sub dir with SNR vs m5 vs cadence files (default: SNR_m5)
    dirSNR_opti: str, opt
       sub dir with SNR opti file (default: SNR_opti)
    dirNvisits: str, opt
      sub dir where outputs will be saved (default: Nvisits_z)
    outName: str,opt
      output file Name
    """

    # load SNR_opti file
    snr_opti_df = pd.DataFrame(
        np.load('{}/{}/{}'.format(dirStudy, dirSNR_opti, snr_opti_file), allow_pickle=True))

    cadvis = Visits_Cadence(snr_opti_df, m5_single)
    bands = np.unique(m5_single['filter'])
    bb = []
    for b in bands:
        bb.append('Nvisits_{}'.format(b))
    # load SNR_m5 files for various cadences
    fis = glob.glob('{}/{}/*'.format(dirStudy, dirSNR_m5))

    res = pd.DataFrame()
    for fi in fis:
        fib = (
            fi.split('/')[-1]
            .split('_')
        )
        idx = fib.index('cad')
        cadence = int(fib[idx+1])
        # load m5 single file
        m5_cad = pd.DataFrame(np.load(fi, allow_pickle=True))
        nv_cad = cadvis(m5_cad)
        nv_cad['cadence'] = cadence
        res = pd.concat((res, nv_cad))

    outFull = '{}/{}/{}'.format(dirStudy, dirNvisits, outName)

    # final check: are all the bands present?
    # if not: add it with one visit
    if 'g' not in res['band'].unique():
        idx = res['band'] == 'i'
        sel = res.loc[idx].copy()
        sel.loc[:, 'band'] = 'g'
        sel.loc[:, 'Nvisits'] = 0.
        sel.loc[:, 'Nvisits_orig'] = 0.
        res = pd.concat((res, sel))

    # replace nan with zeros
    res = res.fillna(0.)
    np.save(outFull, res.to_records(index=False))

    # transform the data to have a format compatible with GUI
    TransformData(res, outFull.split('.npy')[0], grlist=['z', 'cadence'])


class Nvisits_Cadence_Fields:

    def __init__(self, x1=-2.0, color=0.2,
                 error_model=1,
                 errmodrel=-1.,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 dirStudy='dd_design',
                 dirTemplates='Templates',
                 dirNvisits='Nvisits_z',
                 dirm5='m5_files',
                 Nvisits_z_med='Nvisits_med',
                 outName='Nvisits_z_fields',
                 cadences=[1, 2, 3, 4],
                 # min_par=['nvisits','nvisits_sel','nvisits_selb']):
                 cadence_for_opti=0,
                 min_par=['nvisits_selb'],
                 nproc=4):
        """
        class  to estimate the number of visits for DD fields depending on cadence
        from a number of visits defined with median m5 values

        Parameters
        ---------------
        x1 : float, opt
          SN x1 (default: -2.0)
        color: float, opt
          SN color (default : 0.2)
        error_model: int, opt
          error model for LC (default: 1)
        errmodrel: float,opt
          relative flux error model cut (default: -1.)
        bluecutoff: float,opt
          blue cutoff (if error_model=0) (default: 380.)
        redcutoff: float, opt
          red cutoff (if error_model=0) (default: 800.)
        ebvofMW: float, opt
         ebv of MW (default: 0.)
        sn_simulator: str, opt
         simulator to generate the template (default: sn_fast)
        dirStudy: str, opt
          main dir for the study (default: dd_design)
        dirTemplates: str, opt
          subdir of the templates (default: Templates)
        dirNvisits: str, opt
          subdir with the reference file to estimate the results (default: Nvisits_z)
        dirm5: str, opt
          subdir with reference m5 files (default: m5_files)
        Nvisits_z: str, opt
          fileName with reference number of visits vs cadence (default: Nvisits_z_med.npy)
        outName: str, opt
          output file name prefix (default: Nvisits_z_fields)
        cadences: list(int), opt
          list of cadences to process (default: {1,2,3,4])
        cadence_for_opti: int, opt
          cadence used for optimisation (default: 0)
        min_par: list(str), opt
          list on minimization parameters used in SNR_combi (default: ['nvisits','nvisits_sel','nvisits_selb']
        nproc: int, opt
          number of procs for multiprocessing
        """

        self.x1 = x1
        self.color = color
        self.error_model = error_model
        self.errmodrel = errmodrel
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.ebvofMW = ebvofMW
        self.sn_simulator = sn_simulator
        self.dirTemplates = dirTemplates
        self.dirStudy = dirStudy
        self.dirm5 = dirm5
        self.min_par = min_par
        self.cadences = cadences
        self.dirNvisits = dirNvisits
        self.Nvisits_z_med = Nvisits_z_med
        self.cadence_for_opti = cadence_for_opti

        #restot = self.multiproc()

        restot = pd.DataFrame()
        for j, cadence in enumerate(cadences):
            # for j, cadence in enumerate([1]):
            print('cadence', cadence)
            rr = self.nvisits_single_cadence(cadence, nproc)
            restot = pd.concat((restot, rr))

        # replace nan with zeros
        restot = restot.fillna(0.)
        # restot = pd.DataFrame(
        #    np.load('Nvisits_z_fields.npy', allow_pickle=True))
        if self.cadence_for_opti > 0:
            dirNvisits += '_{}'.format(int(self.cadence_for_opti))
        outFull = '{}/{}/{}'.format(dirStudy, dirNvisits, outName)
        TransformData(restot, outFull.split('.npy')[0], grlist=[
            'z', 'cadence', 'fieldname', 'season'])

    def multiproc(self):

        resdf = pd.DataFrame()

        time_ref = time.time()
        result_queue = multiprocessing.Queue()

        #cadences = range(2, 5)
        #cadences = np.unique(self.nvisits_ref['cadence'])
        nproc = len(self.cadences)

        for j, cadence in enumerate(self.cadences):
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.nvisits_single_cadence,
                                        args=(cadence, j, result_queue))
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = pd.DataFrame()

        # gather the results
        for key, vals in resultdict.items():
            restot = pd.concat((restot, vals), sort=False)

        # transformation to get the apprriate format for GUI

        # np.save('Nvisits_z_fields.npy', restot.to_records(index=False))

        print('end of processing', time.time()-time_ref)

        return restot

    def nvisits_single_cadence(self, cadence, nproc=4,
                               j=0, output_q=None):

        # load nvisits_ref
        dirNvisits = '{}_{}'.format(self.dirNvisits, cadence)
        if self.cadence_for_opti > 0:
            dirNvisits = '{}_{}'.format(self.dirNvisits, self.cadence_for_opti)
        refName = '{}/{}/{}'.format(self.dirStudy,
                                    dirNvisits, self.Nvisits_z_med)
        print('loading nvisits_ref', refName)
        nvisits_ref = np.load(refName, allow_pickle=True)

        resdf = pd.DataFrame()
        red = RedshiftLimit(self.x1, self.color,
                            cadence=cadence,
                            error_model=self.error_model,
                            errmodrel=self.errmodrel,
                            bluecutoff=self.bluecutoff, redcutoff=self.redcutoff,
                            ebvofMW=self.ebvofMW,
                            sn_simulator=self.sn_simulator,
                            lcDir='{}/{}'.format(self.dirStudy,
                                                 self.dirTemplates),
                            m5_dir='{}/{}'.format(self.dirStudy, self.dirm5))

        idx = np.abs(nvisits_ref['cadence']-cadence) < 1.e-5
        sela = nvisits_ref[idx]

        min_pars = self.min_par
        if not self.min_par:
            min_pars = np.unique(sela['min_par'])
        for min_par in min_pars:
            print('processing', min_par)
            idb = sela['min_par'] == min_par
            sel_visits = sela[idb]
            respar = red(sel_visits, nproc)
            respar = respar.reset_index(drop=True)
            resdf = pd.concat((resdf, respar))

        if output_q is not None:
            return output_q.put({j: resdf})
        else:
            return resdf

    def multiproc_min_par(self, min_pars, sela, j=0, output_q=None):

        for min_par in min_pars:
            print('processing', min_par)
            idb = sela['min_par'] == min_par
            sel_visits = sela[idb]
            respar = red(sel_visits)
            respar = respar.reset_index(drop=True)
            resdf = pd.concat((resdf, respar))

        if output_q is not None:
            return output_q.put({j: resdf})
        else:
            return resdf


class TransformData:
    """
    class to transform a set of rows to a unique one

    Parameters
    ---------------
    df: pandas df
     data to transform
    fi: str
      npy file to transform
    grlist: list(str)
      used for the groupby df
    """

    def __init__(self, df=None, outName='', grlist=['z', 'cadence']):

        for min_par in np.unique(df['min_par']):
            idx = df['min_par'] == min_par
            sel = df[idx].copy()
            gr = sel.groupby(grlist).apply(
                lambda x: self.transform(x)).reset_index()
            gr['min_par'] = min_par

            if 'season' not in gr.columns:
                gr['season'] = 0
            if 'fieldname' not in gr.columns:
                gr['fieldname'] = 'all'

            if 'zlim' in gr.columns:
                gr['z'] = gr['zlim']
                gr = gr.drop(columns=['zlim'])

            np.save('{}_{}.npy'.format(outName, min_par),
                    gr.to_records(index=False))

    def transform(self, grp):
        """
        Method to transform a set of rows to a unique one

        Parameters
        ---------------
        grp: pandas df group
         data to modify

        Returns
        ----------
        pandas df of the data transformed

        """
        r = []
        names = []

        for b in grp['band'].unique():
            idx = grp['band'] == b
            r.append(grp[idx]['Nvisits'].values.item())
            names.append('Nvisits_{}'.format(b))

        # add the missing bands if necessary
        for b in 'grizy':
            nvisits = 'Nvisits_{}'.format(b)
            if nvisits not in names:
                r.append(0.)
                names.append(nvisits)

        r.append(grp['Nvisits'].sum())
        names.append('Nvisits')

        if 'zlim' in grp.columns:
            r.append(grp['zlim'].median())
            names.append('zlim')

        res = np.rec.fromrecords([r], names=names)

        return pd.DataFrame(res)


class Select_errormodel:
    """
    class to filter LC points according to errormodel cut

    Parameters
    --------------
    theDir: str
      location dir of the files
    simuName: str
      name of the simu file
    errormodelCut: float, opt
      max value of lc[fluxerr_model]/lc[flux]

    """

    def __init__(self, theDir, simuName, errormodelCut=-1.0):

        self.errormodelCut = errormodelCut
        lcName = simuName.replace('Simu', 'LC')
        lcName_new = lcName.replace(
            '.hdf5', '_{}.hdf5'.format(np.round(errormodelCut, 2)))

        simus = loopStack(['{}/{}'.format(theDir, simuName)], 'astropyTable')

        outName = '{}/{}'.format(theDir, lcName_new)

        print('new name', outName)
        for simu in simus:
            lc = Table.read('{}/{}'.format(theDir, lcName),
                            path='lc_{}'.format(simu['index_hdf5']))
            lc_sel = self.select(lc)
            lc_sel.write(outName, 'lc_{}'.format(
                simu['index_hdf5']), append=True, compression=True)

    def select(self, lc):
        """
        function to select LCs

        Parameters
        ---------------
        lc: astropy table
          lc to consider

        Returns
        ----------
        lc with filtered values
       """

        if self.errormodelCut < 0.:
            return lc

        # first: select iyz bands

        bands_to_keep = []

        lc_sel = Table()
        for b in 'izy':
            bands_to_keep.append('LSST::{}'.format(b))
            idx = lc['band'] == 'LSST::{}'.format(b)
            lc_sel = vstack([lc_sel, lc[idx]])

        # now apply selection on g band for z>=0.25
        sel_g = self.sel_band(lc, 'g', 0.25)

        # now apply selection on r band for z>=0.6
        sel_r = self.sel_band(lc, 'r', 0.6)

        lc_sel = vstack([lc_sel, sel_g])
        lc_sel = vstack([lc_sel, sel_r])

        return lc_sel

    def sel_band(self, tab, b, zref):
        """
        Method to performe selections depending on the band and z

        Parameters
        ---------------
        tab: astropy table
          lc to process
        b: str
          band to consider
        zref: float
           redshift below wiwh the cut wwill be applied

        Returns
        ----------
        selected lc
        """

        idx = tab['band'] == 'LSST::{}'.format(b)
        sel = tab[idx]
        if len(sel) == 0:
            return Table()

        if sel.meta['z'] >= zref:
            idb = sel['fluxerr_model']/sel['flux'] <= self.errormodelCut
            selb = sel[idb]
            return selb

        return sel
