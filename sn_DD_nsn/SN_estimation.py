import glob
from sn_tools.sn_io import loopStack
from astropy.table import Table, vstack
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sn_tools.sn_rate import SN_Rate
from scipy.interpolate import interp1d
from optparse import OptionParser
import multiprocessing

def SN(fitDir, dbName, fieldNames, SNtype,nproc=4):
    """
    Method to load files in pandas df

    Parameters
    ----------
    firDir: str
     location dir of the files to load
    dbName: str
      OS name to process
    fieldNames: list(str)
     list of the fields to consider
    SNtype: str
     type of SN to consider
    nproc: int
      number of procs to use (default: 4)

    Returns
    -------
    pandas df of the data


    """
    nfields = int(len(fieldNames))
        
    if nfields <= nproc:
        nproc = nfields

    tabpix = np.linspace(0, nfields, nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()
    nmulti = len(tabpix)-1

    print('there man',tabpix,nmulti)
    for j in range(nmulti):
        ida = tabpix[j]
        idb = tabpix[j+1]
        
        p = multiprocessing.Process(name='Subprocess-'+str(j), target=SN_fields, args=(fitDir, dbName,fieldNames[ida:idb], SNtype, j, result_queue))

        p.start()

    # get the results
    resultdict = {}
    for i in range(nmulti):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = pd.DataFrame()
    # gather the results
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals), sort=False)
    return restot

    
    dfSN = pd.DataFrame()
    for field in fieldNames:
        search_path = '{}/{}/*{}*{}*.hdf5'.format(
            fitDir, dbName, field, SNtype)
        fis = glob.glob(search_path)
        print('result files', fis, search_path)
        out = loopStack(fis, objtype='astropyTable').to_pandas()
        out['fieldName'] = field
        # idx = out['Cov_colorcolor'] >= 1.e-5
        dfSN = pd.concat((dfSN, out))

    return dfSN

def SN_fields(fitDir, dbName, fields, SNtype,j=0, output_q=None):
    """
    Method to load files in pandas df

    Parameters
    ----------
    firDir: str
     location dir of the files to load
    dbName: str
      OS name to process
    fieldNames: list(str)
     list of the fields to consider
    SNtype: str
     type of SN to consider
    j: int
      proc number in multiproc (default: 0)
    output_q: multiprocessing queue
      output queue for multiprocessing (default: None)

    Returns
    -------
    pandas df of the data


    """
    out_fi = pd.DataFrame()

    for field in fields:
        search_path = '{}/{}/*{}*{}*.hdf5'.format(
            fitDir, dbName, field, SNtype)
        fis = glob.glob(search_path)
        print('result files', fis, search_path)
        out = loopStack(fis, objtype='astropyTable').to_pandas()
        out['fieldName'] = field
        out_fi = pd.concat((out_fi,out))
        
    if output_q is not None:
        output_q.put({j: out_fi})
    else:
        return out_fi
     
class SN_zlimit:
    """
    class to estimate observing efficiencies vs redshift

    Parameters
    ---------------
    sn_tot: pandas df
      data to process
    summary: str
     npy file containing a summary of infos related to the OS

    """

    def __init__(self, sn_tot, summary, min_rf_phase_qual=-15., max_rf_phase_qual=30.):

        self.rateSN = SN_Rate(H0=70., Om0=0.3,
                              min_rf_phase=min_rf_phase_qual, max_rf_phase=max_rf_phase_qual)

        self.summary = np.load(summary, allow_pickle=True)
        self.sn_tot = sn_tot
        
    def __call__(self, color_cut=0.04,
                 listNames=['fieldName', 'season', 'pixRA',
                            'pixDec', 'healpixID', 'x1', 'color'],
                 what='zlims', zlims=None, nproc=4):
        """
        Method estimating efficiency vs z for a sigma_color cut

        Parameters
        ---------------
        color_cut: float, opt
         color selection cut (default: 0.04)
        listNames: list(str)
          list of column to be used for the groupby apply
          (default: ['fieldName', 'season', 'pixRA',
                            'pixDec', 'healpixID', 'x1', 'color'])
        what: str, opt
           what to compute (default: zlims)
        zlims: pandas df
          redshift limie values (default: None)

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

        sndf = pd.DataFrame(self.sn_tot)
        # use the multiprocessing depending on fieldNames

        fieldNames = np.unique(self.sn_tot['fieldName'])
        nfields = int(len(fieldNames))
        
        if nfields <= nproc:
            nproc = nfields

        tabpix = np.linspace(0, nfields, nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()
        nmulti = len(tabpix)-1

        print('there man',tabpix,nmulti)
        for j in range(nmulti):
            ida = tabpix[j]
            idb = tabpix[j+1]

            idx = sndf['fieldName'].isin(fieldNames[ida:idb])
            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.process_fields, args=(
                sndf[idx], color_cut,listNames,what, zlims, j, result_queue))

            p.start()

        # get the results
        resultdict = {}
        for i in range(nmulti):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = pd.DataFrame()
        # gather the results
        for key, vals in resultdict.items():
            restot = pd.concat((restot, vals), sort=False)
        return restot
        
        
        
        sndf = pd.DataFrame(self.sn_tot)

        print(sndf.columns)

        # groups = sndf.groupby(listNames)

        # estimating zlims
        resu = None
        if what == 'zlims':
            print(listNames)
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.zlim(x, color_cut)).reset_index(level=list(range(len(listNames))))
        if what == 'nsn':
            """
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.nsn(x, zlims, color_cut)).reset_index(level=list(range(len(listNames))))
            """
            print('listNames', listNames)
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.nsn(x, zlims, color_cut)).reset_index()

        return resu

    def process_fields(self,sndf,color_cut=0.04,
                      listNames=['fieldName', 'season', 'pixRA',
                                 'pixDec', 'healpixID', 'x1', 'color'],
                       what='zlims',zlims=None,
                      j=0, output_q=None):
        """
        Method estimating efficiency vs z for a sigma_color cut
        Parameters
        ---------------
        sndf: pandas df
         data used to estimate efficiencies
        color_cut: float, opt
         color selection cut (default: 0.04)
        listNames: list(str)
          list of column to be used for the groupby apply
          (default: ['fieldName', 'season', 'pixRA',
                            'pixDec', 'healpixID', 'x1', 'color'])
        what: str, opt
           what to compute (default: zlims)
        zlims: pandas df
          redshift limie values (default: None)
        j: int
          proc number (default: 0)
        output_q: multiprocessing_queue
          output_queue for multiprocessing

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

        resu = None
        if what == 'zlims':
            print(listNames)
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.zlim(x, color_cut)).reset_index(level=list(range(len(listNames))))
        if what == 'nsn':
            """
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.nsn(x, zlims, color_cut)).reset_index(level=list(range(len(listNames))))
            """
            print('listNames', listNames)
            resu = sndf.groupby(listNames)[['Cov_colorcolor', 'z', 'survey_area', 'fitstatus']].apply(
                lambda x: self.nsn(x, zlims, color_cut)).reset_index()
            
        if output_q is not None:
            output_q.put({j: resu})
        else:
            return resu
    
    def effiObsdf(self, data, color_cut=0.04, zmin=0.025, zmax=0.8, dz=0.05):
        """
        Method to estimate observing efficiencies for supernovae

        Parameters
        --------------
        data: pandas df - grp
         data to process
        color_cut: float, opt
          selection to apply on sigma_color (default: 0.004)
        zmin: float, opt
          min redshift value (default: 0.025)
        zmax: float, opt
          max redshift value (default: 0.8)
        dz: float, opt
          redshift step (default:0.05)

        Returns
        ----------
        pandas df with the following cols:
        - z: redshift
        - effi: efficiency
        - effi_err: efficiency error
        """

        # reference df to estimate efficiencies
        # df = data.loc[lambda dfa:  np.sqrt(dfa['Cov_colorcolor']) < 100000., :]
        df = data.loc[lambda dfa:  dfa['fitstatus'] == 'fitok', :]

        # selection on sigma_c<= 0.04
        df_sel = df.loc[lambda dfa:  np.sqrt(
            dfa['Cov_colorcolor']) <= color_cut, :]

        # make ratio using histogram
        x1 = df_sel['z'].values
        x2 = df['z'].values
        bins = np.array([0.01]+list(np.arange(zmin, zmax, dz)))
        val_of_bins_x1, edges_of_bins_x1, patches_x1 = plt.hist(
            x1, bins=bins, histtype='step', label="x1")
        val_of_bins_x2, edges_of_bins_x2, patches_x2 = plt.hist(
            x2, bins=bins, histtype='step', label="x2")

        ratio = np.divide(val_of_bins_x1,
                          val_of_bins_x2,
                          where=(val_of_bins_x2 != 0))

        var = [ratio[i]*(1.-ratio[i])/val_of_bins_x2[i]
               if val_of_bins_x2[i] > 0. else 0. for i in range(len(ratio))]
        bins_centered = 0.5*(bins[1:]+bins[:-1])

        effidf = pd.DataFrame({'z': bins_centered,
                               'effi': ratio,
                               'effi_err': np.sqrt(var)})
        # 'effi_var': var})
        effidf = effidf.fillna(0)

        return effidf

    def getSNRate(self, data, zmin, zmax, dz):
        """
        Method to estimate the SN rate

        Parameters
        ----------
        data: pandas df
          data to process - requested to get fieldName, healpixID, ...
        zmin: float, opt
          min redshift value (default: 0.01)
        zmax: float, opt
          max redshift value (default: 0.6)
        dz: float, opt
          redshift step (default:0.01)

        Returns
        -------
        nsn_rate:
        nsn_err_from_rate

        """

        # estimate the expected rate - season length needed for that
        idx = self.summary['fieldName'] == data.name[0]
        idx &= self.summary['healpixID'] == data.name[4]
        idx &= self.summary['season'] == data.name[1]

        season_length = 0.
        cadence = 0.
        if len(self.summary[idx]) > 0:
            season_length = self.summary[idx]['season_length'].item()
            cadence = self.summary[idx]['cadence'].item()
            
        # estimate the rates and nsn vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=zmin,
                                                       zmax=zmax,
                                                       dz=dz,
                                                       duration=season_length,
                                                       survey_area=data['survey_area'].mean(
                                                       ),
                                                       account_for_edges=True)

        """
        # rate interpolation
        rateInterp = interp1d(zz, nsn, kind='linear',
                              bounds_error=False, fill_value=0)
        rateInterp_err = interp1d(zz, err_nsn, kind='linear',
                                  bounds_error=False, fill_value=0)
        """
        return nsn, err_nsn, zz, season_length,cadence

    def zlim(self, data, color_cut=0.04, zmin=0.01, zmax=0.8, dz=0.01, frac=0.95, plot=False):
        """
        Method to estimate the redshift limit
        The principle of the measurement is to convolve the observing efficiency curve
        with the SN rate (vs z). The cumulative nsn(z) is then used to estimate the redshift limit
        (corresponding to frac of SN)

        Parameters
        ----------
        data: pandas df
          data to process (SN)
        color_cut: float, opt
          selection to apply on sigma_color (default: 0.004)
        zmin: float, opt
          min redshift value (default: 0.01)
        zmax: float, opt
          max redshift value (default: 0.8)
        dz: float, opt
          redshift step (default:0.01)

        """

        zplot = list(np.arange(zmin, zmax, dz))
        # get efficiencies
        effidf = self.effiObsdf(data, color_cut=color_cut)

        effiInterp = interp1d(effidf['z'].values, effidf['effi'].values, kind='linear',
                              bounds_error=False, fill_value=0)
        effiInterp_err = interp1d(
            effidf['z'], effidf['effi_err'], kind='linear', bounds_error=False, fill_value=0.)

        # get thr rates here

        nsn_from_rate, nsn_err_from_rate, zz, season_length,cadence = self.getSNRate(
            data, zmin, zmax, dz)

        nsn_cum = np.cumsum(effiInterp(zz)*nsn_from_rate)
        nsn_cum_err = []
        for i in range(len(zz)):
            siga = effiInterp_err(zz[:i+1])*nsn_from_rate[:i+1]
            sigb = effiInterp(zz[:i+1])*nsn_err_from_rate[:i+1]
            nsn_cum_err.append(np.cumsum(
                np.sqrt(np.sum(siga**2 + sigb**2))).item())

        if nsn_cum[-1] >= 1.e-5:
            nsn_cum_norm = nsn_cum/nsn_cum[-1]  # normalize
            nsn_cum_norm_err = nsn_cum_err/nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zz,
                            bounds_error=False, fill_value=-1.)
            zlim_plus = interp1d(nsn_cum_norm+nsn_cum_norm_err,
                                 zz, bounds_error=False, fill_value=-1.)
            zlim_minus = interp1d(
                nsn_cum_norm-nsn_cum_norm_err, zz, bounds_error=False, fill_value=-1.)
            zlimit = zlim(frac)
            zlimit_minus = zlim_plus(frac)
            zlimit_plus = zlim_minus(frac)
        else:
            zlimit = 0.0
            zlimit_plus = 0.0
            zlimit_minus = 0.0

        # print('nsn_cum',nsn_cum[-1],zlimit,zlimit_plus,zlimit_minus)

        if plot:
            self.plotAll(zplot, effidf, nsn_cum,
                         nsn_cum_norm, nsn_cum_norm_err, frac)

        return pd.DataFrame({'zlim': [np.round(zlimit, 2)],
                             'zlim_plus': [np.round(zlimit_plus, 2)],
                             'zlim_minus': [np.round(zlimit_minus, 2)],
                             'season_length': [np.round(season_length, 2)],
                             'cadence': [np.round(season_length, 2)]})
    
    def nsn(self, data, zlimits, color_cut=0.04, zmin=0.01, dz=0.005, plot=False):
        """
        Method to estimate the number of supernovae corresponding with z<= zlim
        The principle of the measurement is to convolve the observing efficiency curve
        with the SN rate (vs z). The cumulative nsn(z) is then used to estimate the redshift limit
        (corresponding to frac of SN)

        Parameters
        ----------
        data: pandas df
          data to process (SN)
        color_cut: float, opt
          selection to apply on sigma_color (default: 0.004)
        zlimits: pandas df
          redshift limits to estimate nsn
        zmin: float, opt
          min redshift value (default: 0.01)
        zmax: float, opt
          max redshift value (default: 0.8)
        dz: float, opt
          redshift step (default:0.01)

        """
        idx = zlimits['fieldName'] == data.name[0]
        idx &= zlimits['season'] == data.name[1]
        idx &= zlimits['healpixID'] == data.name[4]

        zmax = 0.
        if len(zlimits[idx]) > 0:
            zmax = zlimits[idx]['zlim'].item()

        if zmax > 0.01:
            zplot = list(np.arange(zmin, zmax, dz))
            # get efficiencies
            effidf = self.effiObsdf(data, color_cut=color_cut)

            effiInterp = interp1d(effidf['z'].values, effidf['effi'].values, kind='linear',
                                  bounds_error=False, fill_value=0)
            effiInterp_err = interp1d(
                effidf['z'], effidf['effi_err'], kind='linear', bounds_error=False, fill_value=0.)

            # get the rates here

            nsn_from_rate, nsn_err_from_rate, zz, season_length,cadence = self.getSNRate(
                data, zmin, zmax, dz)

            nsn_cum = np.cumsum(effiInterp(zz)*nsn_from_rate)
            nsn_cum_err = []
            for i in range(len(zz)):
                siga = effiInterp_err(zz[:i+1])*nsn_from_rate[:i+1]
                sigb = effiInterp(zz[:i+1])*nsn_err_from_rate[:i+1]
                nsn_cum_err.append(np.cumsum(
                    np.sqrt(np.sum(siga**2 + sigb**2))).item())

            nsn = nsn_cum[-1]
            nsn_err = nsn_cum_err[-1]
        else:
            nsn = 0.
            nsn_err = 0.
            season_length = 0.
            cadence = 0.

        if plot:
            self.plotnSN(zplot, effidf, nsn_cum)

        return pd.DataFrame({'nsn': [np.round(nsn, 2)],
                             'nsn_err': [np.round(nsn_err, 2)],
                             'season_length': [np.round(season_length, 2)],
                             'cadence': [np.round(cadence, 2)],
                             'zlim': [np.round(zmax, 2)]})

    def plotAll(self, zplot, effidf, nsn_cum, nsn_cum_norm, nsn_cum_norm_err, frac):
        """
        Method to plot efficiency, nsn cum, ... vs z

        Parameters
        ----------
        zplot: list
          redshift values
        effidf: pandas df
          efficiencies
        nsn_cum: array
         cumulative number of sn
        nsn_cum_norm: array
         normalized cumulative number of sn
        nsn_cum_norm_err: array
         error of the normalized cumulative number of sn
        frac: float
         fraction of SN used to estimate zlim

        """

        # this is to display results
        fig, ax = plt.subplots()
        # plot efficiencies
        ax.errorbar(effidf['z'], effidf['effi'], yerr=effidf['effi_err'])
        # plot rate (cumul)
        axb = ax.twinx()
        axb.plot(zplot, nsn_cum)
        # plot cumul of nsn*effi
        ax.plot(zplot, nsn_cum/nsn_cum[-1])
        ax.plot(zplot, [frac]*len(zplot), color='r')
        ax.fill_between(zplot, nsn_cum_norm-nsn_cum_norm_err,
                        nsn_cum_norm+nsn_cum_norm_err, color='y')
        plt.show()

    def plotnSN(self, zplot, effidf, nsn_cum):
        """
        Method to plot efficiency, nsn cum, ... vs z

        Parameters
        ----------
        zplot: list
          redshift values
        effidf: pandas df
          efficiencies
        nsn_cum: array
         cumulative number of sn
        """

        # this is to display results
        fig, ax = plt.subplots()
        # plot efficiencies
        ax.errorbar(effidf['z'], effidf['effi'], yerr=effidf['effi_err'])
        # plot rate (cumul)
        axb = ax.twinx()
        axb.plot(zplot, nsn_cum)
        # plot cumul of nsn*effi
        ax.plot(zplot, nsn_cum)

        plt.show()


class NSN_zlim:
    """
    class to estimate the number of supernovae up to zlim

    Parameters
    ----------
    data: pandas df
     data (SN) to process
    zlims: pandas df
      redshift limits

    """

    def __init__(self, data, zlims):

        # merge the two dfs
        tomerge = ['fieldName', 'season', 'pixRA', 'pixDec', 'healpixID']
        sn = data.merge(zlims, left_on=tomerge, right_on=tomerge)

        nsn_simu = data.groupby(
            ['fieldName']).size().to_frame('N').reset_index()

        print('allo', nsn_simu)
        sn = sn.groupby(tomerge).apply(lambda x: nSN(x)).reset_index()

        nsn_sum = sn.groupby(['fieldName']).apply(
            lambda x: self.nsn_summary(x, nsn_simu)).reset_index()

        print(nsn_sum)

    def nsn_summary(self, grp, nsn_simu):
        """
        Method to estimate the number of sn for a given group

        Parameters
        ---------------
        grp: pandas df group
         data to process

        Returns
        -----------


        """

        idx = nsn_simu['fieldName'] == grp.name
        N = nsn_simu[idx]['N'].values.item()
        p = np.sum(grp['nsn_zlim_noccut'])/N
        print('ici', grp.name, N)
        return pd.DataFrame({'nsn': [np.sum(grp['nsn_zlim_noccut'])],
                             'err_nsn': [np.sqrt(N*p*(1.-p))],
                             'nsn_simu': [N]})


def pixels(grp):

    return pd.DataFrame({'npixels': [len(np.unique(grp['healpixID']))]})


def finalNums(grp, normfact=10.):

    return pd.DataFrame({'nsn_zlim': [grp['nsn_zlim'].sum()/normfact],
                         'nsn_tot': [grp['nsn_tot'].sum()/normfact],
                         'zlim': [grp['zlim'].median()]})


def nSN(grp, sigmaC=0.04):

    idx = grp['Cov_colorcolor'] <= sigmaC**2
    idx &= grp['fitstatus'] == 'fitok'
    sela = grp[idx]
    idx &= grp['z'] <= grp['zlim']
    sel = grp[idx]

    ido = grp['z'] <= grp['zlim']
    print(grp.name, len(grp[ido]))
    return pd.DataFrame({'nsn_zlim': [len(sel)],
                         'nsn_zlim_noccut': [len(grp[ido])],
                         'nsn_tot': [len(sela)],
                         'zlim': [grp['zlim'].median()]})


def zlim(grp, sigmaC=0.04):

    ic = grp['Cov_colorcolor'] <= sigmaC**2
    selb = grp[ic]

    if len(selb) == 0:
        zl = 0.
    else:
        zl = np.max(selb['z'])

    return pd.DataFrame({'zlim': [zl]})


parser = OptionParser()

parser.add_option("--mainDir", type="str", default='/home/philippe/LSST/DD_Full_Simu_Fit',
                  help="main loc dir of the files (mainDir/Fit) [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs', help="db name [%default]")
parser.add_option("--fieldNames", type=str, default='COSMOS,CDFS,XMM-LSS,ELAIS,ADFS1,ADFS2',
                  help="fieldNames [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="number of proc to use for multiprocessing[%default]")

opts, args = parser.parse_args()

mainDir = opts.mainDir
dbName = opts.dbName
fieldNames = opts.fieldNames.split(',')
nproc = opts.nproc

fitDir = '{}/Fit'.format(mainDir)
# fitDir = 'OutputFit'
simuDir = '{}/Simu'.format(mainDir)


allSN = pd.DataFrame()
zlimit = None

summary = 'DD_Summary_{}.npy'.format(dbName)
# get faintSN data
faintSN = SN(fitDir, dbName, fieldNames, 'faintSN',nproc=nproc)

# estimate the redshift limits for faint SN
SN_zlims_faint = SN_zlimit(faintSN, summary)

zlims_faint = SN_zlims_faint()

print(zlims_faint)

print('zlimit faint', zlims_faint.groupby(['fieldName'])['zlim'].median())

# load all types of SN
allSN = SN(fitDir, dbName, fieldNames, 'allSN',nproc=nproc)
print('eee', allSN[['x1', 'color']])

SN_nSN_all = SN_zlimit(allSN, summary)
# estimate the number of supernovae
#nsn = NSN_zlim(allSN, zlims_faint)

nsn_zlim = SN_nSN_all(listNames=['fieldName', 'season', 'pixRA',
                                 'pixDec', 'healpixID'],
                      what='nsn', zlims=zlims_faint,nproc=nproc)

print('go west', nsn_zlim)
print(nsn_zlim.groupby(['fieldName'])['nsn'].sum())

outName = 'nSN_zlim_DD_{}.npy'.format(dbName)

# add db Name in output numpy array
nsn_zlim['dbName'] = dbName
np.save(outName, nsn_zlim.to_records(index=False))
