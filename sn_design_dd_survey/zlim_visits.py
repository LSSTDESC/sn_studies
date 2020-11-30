import os
import h5py
from astropy.table import Table, vstack
import numpy as np


class RedshiftLimit:
    """
    class to estimate the redshift limit for a SN
    using fast simulation

    Parameters
    ---------------
    x1: float,opt
      SN strech (default: -2.0)
    color: float
      SN color (default: 0.2)
    Nvisits: dict, opt
      number of visits for each band (default: 'grizy',[10,20,20,26,20])
    m5: dict, opt
      fiveSigmaDepth single visit for each band (default: 'grizy',[24.51,24.06,23.62,23.0,22.17])
    cadence: dict, opt
      cadence of observation (per band) (default='grizy',[3.,3.,3.,3.3.])
    error_model: int, opt
      to use error model or not (default: 1)
    bluecutoff: float, opt
      blue cutoff to apply (if error_model=0) (default: 380.)
    redcutoff: float, opt
      red cutoff to apply (if error_model=0) (default: 800.)
    simulator: str, opt
      simulator to use (default: sn_fast)
    fitter: str, opt
      fitter to use (defaulf: sn_fast)
    tag: str, opt
      tag for the production (default: test)

    """

    def __init__(self, x1=-2.0, color=0.2,
                 Nvisits=dict(zip('grizy', [10, 20, 20, 26, 20])),
                 m5=dict(zip('grizy', [24.51, 24.06, 23.62, 23.0, 22.17])),
                 cadence=dict(zip('grizy', [3., 3., 3., 3., 3.])),
                 error_model=1,
                 bluecutoff=380.,
                 redcutoff=800.,
                 simulator='sn_fast',
                 fitter='sn_fast',
                 tag='test'):

        self.x1 = x1
        self.color = color
        self.Nvisits = Nvisits
        self.m5 = m5
        self.cadence = cadence
        self.bands = self.Nvisits.keys()
        self.tag = tag

        # simulation parameters
        self.error_model = error_model
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

    def process(self):
        """
        Method to process data in three steps:
        - generate fake obs
        - generate LCs from fake obs
        - fit LC generated from fake obs

        """

        # generate observations
        self.generate_obs()

        # simulation(fast) of LCs
        self.simulate_lc()

        # fit (fast) these LCs
        self.fit_lc()

    def generate_obs(self):
        """
        Method to generate fake observations

        """
        # generate fake_config file
        cmd = 'python run_scripts/make_yaml/make_yaml_fakes.py'
        for b in self.bands:
            cmd += ' --cadence_{} {}'.format(b, self.cadence[b])
            cmd += ' --m5_{} {}'.format(b, self.m5[b])
            cmd += ' --Nvisits_{} {}'.format(b, self.Nvisits[b])
        cmd += ' --fileName {}'.format(self.fake_config)
        os.system(cmd)

        # create fake data from yaml configuration file
        cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
            self.fake_config, self.fake_data)
        os.system(cmd)

    def simulate_lc(self):
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
        cmd += ' --SN_z_min {}'.format(self.zmin)
        cmd += ' --SN_z_max {}'.format(self.zmax)
        cmd += ' --SN_z_step {}'.format(self.zstep)
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

    def zlim(self, color_cut=0.04):
        """
        Method to estimate the redshift limit

        Parameters
        ---------------
        color_cut: float, opt
           sigmaColor cut (default: 0.04)

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
