import os
import h5py
from astropy.table import Table, vstack
import numpy as np
from sn_tools.sn_io import loopStack
import pandas as pd
from sn_tools.sn_calcFast import CovColor
from scipy.interpolate import interp1d
import multiprocessing


class zlim_template:

    def __init__(self, x1, color, cadence,  error_model=1,
                 errmodrel=-1.,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 lcDir='.',
                 m5_file='medValues_flexddf_v1.4_10yrs_DD.npy',
                 m5_dir='dd_design/m5_files',
                 include_error_model_sigmac=False):

        cutof = self.cutoff(error_model, bluecutoff, redcutoff)
        lcName = 'LC_{}_{}_{}_ebv_{}_{}_cad_{}_0.hdf5'.format(
            sn_simulator, x1, color, ebvofMW, cutof, int(cadence))
        self.lcFullName = '{}/{}'.format(lcDir, lcName)

        simuFullName = self.lcFullName.replace('LC', 'Simu')
        self.lc_meta = loopStack([simuFullName], objtype='astropyTable')
        self.lc_meta.convert_bytestring_to_unicode()

        self.Fisher_el = ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                          'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor', 'F_colorcolor']
        self.errmodrel = errmodrel
        self.include_error_model_sigmac=include_error_model_sigmac
        
    def cutoff(self, error_model, bluecutoff, redcutoff):

        cuto = '{}_{}'.format(bluecutoff, redcutoff)
        if error_model:
            cuto = 'error_model'

        return cuto

    def process(self, zmin, zmax, zstep, nvisits, m5_values, nproc):

        r = []
        self.lc_visits = {}
        m5_values['band'] = 'LSST::' + m5_values['band'].astype(str)

        zvals = list(np.arange(zmin, zmax, zstep))
        nz = len(zvals)
        t = np.linspace(0, nz, nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.process_zrange,
                                         args=(zvals[t[j]:t[j+1]], m5_values, j, result_queue))
                 for j in range(nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = []

        # gather the results
        for key, vals in resultdict.items():
            restot += vals

        res = np.rec.fromrecords(restot, names=['z', 'sigmaC'])
        """
        import matplotlib.pyplot as plt
        plt.plot(res['z'],res['sigmaC'])
        plt.show()
        """
        zlim = self.estimate_zlim(res)

        return zlim

    """
        for zval in np.arange(zmin, zmax, zstep):
            lc = self.getLC(zval)
            lc.convert_bytestring_to_unicode()
            lcc = self. lc_corr(lc, m5_values)
            self.lc_visits[np.round(zval, 2)] = lcc
            sigmaC = self.fit(lcc, m5_values[['band', 'm5_new']])
            r.append((zval, sigmaC))

        res = np.rec.fromrecords(r, names=['z', 'sigmaC'])

        zlim = self.estimate_zlim(res)

   
        # print('zlim',zlim)

        return zlim
    """

    def process_zrange(self, zvals, m5_values, j=0, output_q=None):

        r = []
        for zval in zvals:
            lc = self.getLC(zval)
            lc.convert_bytestring_to_unicode()
            lcc = self. lc_corr(lc, m5_values)
            self.lc_visits[np.round(zval, 2)] = lcc
            sigmaC = self.fit(lcc, m5_values[['band', 'm5_new']])
            r.append((zval, sigmaC))

        if output_q is not None:
            return output_q.put({j: r})
        else:
            return resdf

    def getLC(self, z):

        idx = np.abs(self.lc_meta['z']-z) < 1.e-8
        sel = self.lc_meta[idx]

        lc = Table.read(self.lcFullName, path='lc_{}'.format(
            sel['index_hdf5'].item()))

        lc['phase'] = (lc['time']-lc.meta['daymax'])/(1.+z)
        return lc

    def fit(self, lc, m5_values):

        idx = lc['flux'] > 0.
        if not self.include_error_model_sigmac:
            lc['fluxerr'] =lc['fluxerr_photo']
            
        idx &= lc['fluxerr'] > 0.

        #selecta = lc.loc[idx, :]
        selecta = lc[idx]
        # add snr
        """
        selecta.loc[:, 'snr'] = selecta['flux'].values / \
            selecta['fluxerr'].values
        """
        selecta['snr'] = selecta['flux']/selecta['fluxerr']
        # select LC points according to SNRmin
        idx = selecta['snr'] >= 1.
        #selecta = selecta.loc[idx, :]
        selecta = selecta[idx]

        if self.errmodrel:
            selecta = self.select_error_model(selecta)

        # fit here
        covcolor = CovColor(selecta.to_pandas()[
                            self.Fisher_el].sum()).Cov_colorcolor
        sigmaC = np.sqrt(covcolor)

        return sigmaC

    def estimate_zlim(self, tab, sigmaC_cut=0.04):
        """
        interp = interp1d(tab['sigmaC'], tab['z'],
                          fill_value=0.0, bounds_error=False)

        return interp(sigmaC_cut).item()
        """
        tab.sort(order='z')
        interpv = interp1d(tab['z'], tab['sigmaC'],
                           bounds_error=False, fill_value=0.)

        zvals = np.arange(0.1, 1.0, 0.005)

        colors = interpv(zvals)
        ii = np.argmin(np.abs(colors-sigmaC_cut))

        return np.round(zvals[ii], 3)

    def lc_corr(self, lc, m5_values):

        # print('io', lc, m5_values)
        df = pd.DataFrame(np.copy(lc))
        df = df.merge(m5_values, left_on=['band'], right_on=['band'])

        # correct LC quantities
        df['fluxerr_old'] = df['fluxerr']
        # first: photometric error
        df['fluxerr_photo'] = df['fluxerr_photo'] * \
            10**(-0.4*(df['m5_new']-df['m5']))
        # second: fluxerr
        df['fluxerr'] = np.sqrt(df['fluxerr_photo']**2+df['fluxerr_model']**2)

        # finally: Fisher elements
        for vv in self.Fisher_el:
            df[vv] = df[vv] * (df['fluxerr_old']/df['fluxerr'])**2

        idx = df['m5_new'] < 0.1
        df.loc[idx, 'flux'] = 0.0
        df.loc[idx, 'flux_e_sec'] = 0.0
        df.loc[idx, 'fluxerr'] = -10.

        tab = Table.from_pandas(df)
        tab.meta = lc.meta
        return tab

    def select_error_model(self, lc):
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

        if self.errmodrel < 0.:
            return lc

        # first: select iyz bands

        bands_to_keep = []

        lc_sel = Table()
        for b in 'izy':
            bands_to_keep.append('LSST::{}'.format(b))
            idx = lc['band'] == 'LSST::{}'.format(b)
            lc_sel = vstack([lc_sel, lc[idx]])

        # now apply selection on g band for z>=0.25
        sel_g = self.sel_band(lc, 'g', 0.35)

        # now apply selection on r band for z>=0.6
        sel_r = self.sel_band(lc, 'r', 0.65)

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
            idb = sel['fluxerr_model']/sel['flux'] <= self.errmodrel
            selb = sel[idb]
            return selb

        return sel


class RedshiftLimit:
    """
    class to estimate the redshift limit for a SN
    using LC template
    """

    def __init__(self, x1, color, cadence,  error_model=1,
                 errmodrel=-1.,
                 bluecutoff=380., redcutoff=800.,
                 ebvofMW=0.,
                 sn_simulator='sn_fast',
                 lcDir='.',
                 m5_file='medValues_flexddf_v1.4_10yrs_DD.npy',
                 m5_dir='dd_design/m5_files',
                 check_fullsim=False):

        m5_file = pd.DataFrame(
            np.load('{}/{}'.format(m5_dir, m5_file), allow_pickle=True))

        m5_file = m5_file.groupby(
            ['fieldname', 'filter', 'season']).median().reset_index()

        m5_file = m5_file[['fieldname', 'filter', 'fiveSigmaDepth', 'season']]

        self.m5_file = m5_file.rename(
            columns={'filter': 'band', 'fiveSigmaDepth': 'm5_single'})

        """
        idx = self.m5_file['fieldname']=='COSMOS'
        idx &= self.m5_file['season']==1
        self.m5_file = self.m5_file[idx]
        """
        self.Fisher_el = ['F_x0x0', 'F_x0x1', 'F_x0daymax', 'F_x0color', 'F_x1x1',
                          'F_x1daymax', 'F_x1color', 'F_daymaxdaymax', 'F_daymaxcolor', 'F_colorcolor']

        self.check_fullsim = check_fullsim

        if check_fullsim:
            self.full_sim = zlim_sim(simulator='sn_cosmo', fitter='sn_cosmo',
                                     error_model=error_model, bluecutoff=bluecutoff)
        self.templ_sim = zlim_template(x1, color,
                                       cadence,  error_model=error_model,
                                       errmodrel=errmodrel,
                                       bluecutoff=bluecutoff, redcutoff=redcutoff,
                                       ebvofMW=ebvofMW,
                                       sn_simulator=sn_simulator,
                                       lcDir=lcDir,
                                       m5_file=m5_file,
                                       m5_dir=m5_dir)

    def __call__(self, nvisits_ref, nproc):
        """
        resdf = pd.DataFrame()
        for field in self.m5_file['fieldname']:
            idx = self.m5_file['fieldname'] == field
            print('field', field)
            resfield = self.zlim(nvisits_ref, self.m5_file.loc[idx, :])
            resfield['field'] = field
            resdf = pd.concat((resdf, resfield))
        """
        resdf = self.m5_file.groupby(['fieldname', 'season']).apply(
            lambda x: self.zlim(nvisits_ref, x, nproc)).reset_index()
        return resdf

    def zlim(self, nvisits_ref, m5_single, nproc):

        rs = []
        resdf = pd.DataFrame()
        print('m5', m5_single)

        for z in np.unique(nvisits_ref['z']):
            idx = np.abs(nvisits_ref['z']-z) < 1.e-6
            nvisits_z = pd.DataFrame(np.copy(nvisits_ref[idx]))

            m5_values = self.m5(nvisits_z, m5_single)
            # print('m5_values',z,m5_values)
            dict_visits = dict(zip(nvisits_z['band'], nvisits_z['Nvisits']))
            dict_m5 = dict(zip(m5_values['band'], m5_values['m5_single']))
            dict_cadence = dict(zip(m5_values['band'], m5_values['cadence']))
            zmin = np.max([0.3, z-0.2])
            zmax = np.min([1., z+0.3])
            zstep = 0.05
            zlimit = self.templ_sim.process(
                zmin, zmax, zstep, nvisits_z, m5_values, nproc)
            nvisits_z['zlim'] = np.round(zlimit, 2)
            resdf = pd.concat((resdf, nvisits_z))
            if self.check_fullsim:
                self.full_sim.process(dict_visits, dict_m5,
                                      dict_cadence, zmin, zmax, zstep)
                zlimit_sim = self.full_sim.zlim()
                rs.append((z, zlimit, zlimit_sim))
            """
            lc_sim = self.full_sim.getLC(z)
            lc_sim['snr'] = lc_sim['flux']/lc_sim['fluxerr']
            print(z, lc_sim['band', 'flux_e_sec', 'snr', 'time'])
            """

            #lc_templ = self.templ_sim.lc_visits[np.round(z, 2)]

            #self.plot(lc_templ, lc_sim)

        if self.check_fullsim:
            print(rs)

        if 'level_2' in resdf.columns:
            resdf = resdf.drop(columns=['level_2'])
        print(resdf.columns)
        return resdf

    def m5(self, nvisits, m5_single):

        df = pd.DataFrame(nvisits)
        df = df.merge(m5_single, left_on=['band'], right_on=['band'])
        df = df.round({'Nvisits': 0})
        df['m5_new'] = df.apply(
            lambda x: x['m5_single']+1.25*np.log10(x['Nvisits']) if x['Nvisits'] > 0 else 0, axis=1)

        return df

    def plot(self, lca, lcb):

        import matplotlib.pyplot as plt
        print(np.unique(lca['band']), np.unique(lcb['band']))

        bands = 'ugrizy'

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
        loc = dict(
            zip(bands, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))

        for band in bands:
            iloc = loc[band][0]
            jloc = loc[band][1]
            idx = lca['band'] == 'LSST::'+band
            if len(lca[idx]) > 0:
                print(band, lca[idx]
                      [['flux_e_sec', 'm5', 'm5_new', 'fluxerr', 'fluxerr_photo', 'fluxerr_model']])
                ax[iloc, jloc].plot(
                    lca[idx]['phase'], lca[idx]['flux_e_sec'], 'ko', mfc='None')
            idx = lcb['band'] == 'LSST::'+band
            if len(lcb[idx]) > 0:
                print('bbb', band, lcb[idx][[
                      'flux_e_sec', 'm5', 'fluxerr', 'fluxerr_photo', 'fluxerr_model']])
                ax[iloc, jloc].plot(lcb[idx]['phase'],
                                    lcb[idx]['flux_e_sec'], 'r*')

        plt.show()


class zlim_sim:
    """
    class to estimate the redshift limit for a SN
    using fast simulation and scripts

    Parameters
    ---------------
    x1: float, opt
      SN strech(default: -2.0)
    color: float
      SN color(default: 0.2)
    Nvisits: dict, opt
      number of visits for each band(default: 'grizy', [10, 20, 20, 26, 20])
    m5: dict, opt
      fiveSigmaDepth single visit for each band(default: 'grizy', [24.51, 24.06, 23.62, 23.0, 22.17])
    cadence: dict, opt
      cadence of observation(per band)(default='grizy', [3., 3., 3., 3.3.])
    error_model: int, opt
      to use error model or not (default: 1)
    bluecutoff: float, opt
      blue cutoff to apply(if error_model=0)(default: 380.)
    redcutoff: float, opt
      red cutoff to apply(if error_model=0)(default: 800.)
    simulator: str, opt
      simulator to use(default: sn_fast)
    fitter: str, opt
      fitter to use(defaulf: sn_fast)
    tag: str, opt
      tag for the production(default: test)

    """

    def __init__(self, x1=-2.0, color=0.2,
                 error_model=1,
                 errmodrel=-1.,
                 bluecutoff=380.,
                 redcutoff=800.,
                 simulator='sn_fast',
                 fitter='sn_fast',
                 tag='test'):

        self.x1 = x1
        self.color = color
        self.tag = tag

        # simulation parameters
        self.error_model = error_model
        self.errmodrel = errmodrel
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        if self.error_model:
            self.cutoff = 'error_model'
        else:
            self.cutoff = '{}_{}'.format(self.bluecutoff, self.redcutoff)
        self.ebvofMW = 0.
        self.simulator = simulator
        self.zmin = 0.01
        self.zmax = 1.0
        self.zstep = 0.05

        # fit parameters
        self.snrmin = 1.0
        self.nbef = 4
        self.naft = 10
        self.nbands = 0
        self.fitter = fitter

        # define some directory for the production of LC+Simu

        self.outDir_simu = 'zlim_simu'
        self.outDir_fit = 'zlim_fit'
        self.outDir_obs = 'zlim_obs'

        # check whether output dir exist - create if necessary
        self.check_create(self.outDir_simu)
        self.check_create(self.outDir_fit)
        self.check_create(self.outDir_obs)

        self.fake_name = 'Fakes_{}'.format(self.tag)
        self.fake_config = '{}/fake_config_{}.yaml'.format(
            self.outDir_obs, tag)
        self.fake_data = '{}/{}'.format(self.outDir_obs, self.fake_name)

    def check_create(self, dirname):
        """
        Method to create a dir if it does not exist

        Parameters
        ---------------
        dirname: str
          directory name

        """
        if not os.path.exists(dirname):
            os.system('mkdir -p {}'.format(dirname))

    def process(self, Nvisits=dict(zip('grizy', [10, 20, 20, 26, 20])),
                m5=dict(zip('grizy', [24.51, 24.06, 23.62, 23.0, 22.17])),
                cadence=dict(zip('grizy', [3., 3., 3., 3., 3.])),
                zmin=0.01, zmax=0.9, zstep=0.05):
        """
        Method to process data in three steps:
        - generate fake obs
        - generate LCs from fake obs
        - fit LC generated from fake obs

        """

        # generate observations
        self.generate_obs(Nvisits, m5, cadence)

        # simulation(fast) of LCs
        self.simulate_lc(zmin, zmax, zstep)

        # fit (fast) these LCs
        self.fit_lc()

    def generate_obs(self, Nvisits, m5, cadence):
        """
        Method to generate fake observations

        """

        bands = Nvisits.keys()
        # generate fake_config file
        cmd = 'python run_scripts/make_yaml/make_yaml_fakes.py'
        for b in bands:
            cmd += ' --cadence_{} {}'.format(b, cadence[b])
            cmd += ' --m5_{} {}'.format(b, m5[b])
            cmd += ' --Nvisits_{} {}'.format(b, int(np.round(Nvisits[b], 0)))
        cmd += ' --fileName {}'.format(self.fake_config)
        os.system(cmd)

        # create fake data from yaml configuration file
        cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
            self.fake_config, self.fake_data)
        os.system(cmd)

    def simulate_lc(self, zmin, zmax, zstep):
        """
        Method to simulate LC

        """
        cmd = 'python run_scripts/simulation/run_simulation.py --dbDir .'
        cmd += ' --dbDir {}'.format(self.outDir_obs)
        cmd += ' --dbName {}'.format(self.fake_name)
        cmd += ' --dbExtens npy'
        cmd += ' --SN_x1_type unique'
        cmd += ' --SN_x1_min {}'.format(self.x1)
        cmd += ' --SN_color_type unique'
        cmd += ' --SN_color_min {}'.format(self.color)
        cmd += ' --SN_z_type uniform'
        cmd += ' --SN_z_min {}'.format(zmin)
        cmd += ' --SN_z_max {}'.format(zmax)
        cmd += ' --SN_z_step {}'.format(zstep)
        cmd += ' --SN_daymax_type unique'
        cmd += ' --Observations_fieldtype Fake'
        cmd += ' --Observations_coadd 0'
        cmd += ' --radius 0.01'
        cmd += ' --Output_directory {}'.format(self.outDir_simu)
        cmd += ' --Simulator_name sn_simulator.{}'.format(self.simulator)
        cmd += ' --Multiprocessing_nproc 1'
        cmd += ' --RAmin 0.0'
        cmd += ' --RAmax 0.1'
        cmd += '  --ProductionID {}'.format(self.tag)
        cmd += ' --SN_blueCutoff {}'.format(self.bluecutoff)
        cmd += ' --SN_redCutoff {}'.format(self.redcutoff)
        cmd += ' --SN_ebvofMW {}'.format(self.ebvofMW)
        cmd += ' --npixels -1'
        cmd += ' --Simulator_errorModel {}'.format(self.error_model)

        os.system(cmd)

    def fit_lc(self):
        """
        Method to fit light curves

        """
        cmd = 'python run_scripts/fit_sn/run_sn_fit.py'
        cmd += ' --Simulations_dirname {}'.format(self.outDir_simu)
        cmd += ' --Simulations_prodid {}_0'.format(self.tag)
        cmd += ' --mbcov_estimate 0 --Multiprocessing_nproc 1'
        cmd += ' --Output_directory {}'.format(self.outDir_fit)
        cmd += ' --LCSelection_snrmin {}'.format(self.snrmin)
        cmd += ' --LCSelection_nbef {}'.format(self.nbef)
        cmd += ' --LCSelection_naft {}'.format(self.naft)
        cmd += ' --LCSelection_nbands {}'.format(self.nbands)
        cmd += ' --Fitter_name sn_fitter.fit_{}'.format(self.fitter)
        cmd += ' --ProductionID {}_{}'.format(self.tag, self.fitter)
        cmd += ' --LCSelection_errmodrel {}'.format(self.errmod)
        os.system(cmd)

    def getSN(self):
        """
        Method to load SN from file

        Returns
        -----------
        sn: astropy table

        """
        fName = '{}/Fit_{}_{}.hdf5'.format(self.outDir_fit,
                                           self.tag, self.fitter)
        fFile = h5py.File(fName, 'r')
        keys = list(fFile.keys())
        sn = Table()
        for key in keys:
            tab = Table.read(fName, path=key)
            sn = vstack([sn, tab])

        return sn

    def getLC(self, z=0.5):
        """"
        Method to load the light curve from file

        Returns
        -----------
        the light curve (format: astropyTable)

        """
        fName = '{}/LC_{}_0.hdf5'.format(self.outDir_simu,
                                         self.tag)
        fFile = h5py.File(fName, 'r')
        keys = list(fFile.keys())
        lc = Table()
        for key in keys:
            tab = Table.read(fName, path=key)
            if np.abs(tab.meta['z']-z) < 1.e-8:
                lc = vstack([lc, tab])

        return lc

    def zlim(self, color_cut=0.04):
        """
        Method to estimate the redshift limit

        Parameters
        ---------------
        color_cut: float, opt
           sigmaColor cut(default: 0.04)

        """

        if 'sn' not in globals():
            self.sn = self.getSN()

        # make interpolation
        from scipy.interpolate import interp1d
        interp = interp1d(np.sqrt(self.sn['Cov_colorcolor']),
                          self.sn['z'], bounds_error=False, fill_value=0.)

        zlim = np.round(interp(color_cut), 2)
        return zlim

    def plot(self, color_cut=0.04):
        """
        Method to plot sigmaC vs z

        Parameters
        ---------------
        color_cut: float, opt
          sigmaColor value to plot(default: 0.04)

        """

        if 'sn' not in globals():
            self.sn = self.getSN()

        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        zlim = self.zlim(color_cut)
        fig, ax = plt.subplots()
        ax.plot(self.sn['z'], np.sqrt(self.sn['Cov_colorcolor']),
                label='zlim={}'.format(zlim), color='r')

        ax.plot(ax.get_xlim(), [color_cut]*2,
                linestyle='--', color='k')

        zmin = 0.2
        zmax = zlim+0.1
        interp = interp1d(self.sn['z'], np.sqrt(self.sn['Cov_colorcolor']),
                          bounds_error=False, fill_value=0.)
        ax.set_xlim([zmin, zmax])
        ax.set_ylim([interp(zmin), interp(zmax)])
        ax.grid()
        ax.set_xlabel('z')
        ax.set_ylabel('$\sigma_{color}$')
        ax.legend(loc='upper left')
        plt.show()


def plot_sigmaC_z(sn, zlim, color_cut=0.04):
    """
    Method to plot sigmaC vs z

    Parameters
    ---------------
    color_cut: float, opt
    sigmaColor value to plot(default: 0.04)

    """

    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    fig, ax = plt.subplots()
    ax.plot(sn['z'], np.sqrt(sn['Cov_colorcolor']),
            label='zlim={}'.format(zlim), color='r')

    ax.plot(ax.get_xlim(), [color_cut]*2,
            linestyle='--', color='k')

    zmin = 0.2
    zmax = zlim+0.1
    interp = interp1d(sn['z'], np.sqrt(sn['Cov_colorcolor']),
                      bounds_error=False, fill_value=0.)
    ax.set_xlim([zmin, zmax])
    ax.set_ylim([interp(zmin), interp(zmax)])
    ax.grid()
    ax.set_xlabel('z')
    ax.set_ylabel('$\sigma_{color}$')
    ax.legend(loc='upper left')
    plt.show()
