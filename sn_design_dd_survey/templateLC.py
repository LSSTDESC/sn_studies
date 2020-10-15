import os

def templateLC(x1,color,simulator,ebvofMW,bluecutoff,redcutoff,error_model,fake_config,zmin,zmax,zstep,outDir):
    """
    Method used to simulate LC from Fakes

    Parameters
    ---------------
    x1: float
      SN x1
    color: float
      SN color
    simulator
     simulator to use
    ebv: float
      ebvofMW value
    bluecutoff: float
      blue cutoff for SN
    redcutoff: float
     red cutoff for SN
    error_model: int
      to activate error model for LC points error
    fake_config: str
      reference config file to generate fakes
    zmin:float
      min redshift value for fake generation
    zmax: float
      max redshift value for fake generation
    outDir: str
      output directory

    """
    fake_output = 'Fake_DESC'
    
    cutoff = '{}_{}'.format(bluecutoff,redcutoff)
    if error_model:
        cutoff = 'error_model'
    
    #outDir_simu = 'Output_Simu_{}_ebvofMW_{}'.format(cutoff,ebv)
    outDir_simu = outDir
    prodid = '{}_Fake_{}_seas_-1_{}_{}_{}_ebvofMW_{}'.format(
    simulator, fake_output, x1, color, cutoff, ebvofMW,error_model)

    # first step: create fake data from yaml configuration file
    cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
        fake_config, fake_output)

    os.system(cmd)

    # now run the full simulation on these data

    cmd = 'python run_scripts/simulation/run_simulation.py --dbDir .'
    cmd += ' --dbName {}'.format(fake_output)
    cmd += ' --dbExtens npy'
    cmd += ' --SN_x1_type unique'
    cmd += ' --SN_x1_min {}'.format(x1)
    cmd += ' --SN_color_type unique'
    cmd += ' --SN_color_min {}'.format(color)
    cmd += ' --SN_z_type uniform'
    cmd += ' --SN_z_min {}'.format(zmin)
    cmd += ' --SN_z_max {}'.format(zmax)
    cmd += ' --SN_z_step {}'.format(zstep)
    cmd += ' --SN_daymax_type unique'
    cmd += ' --Observations_fieldtype Fake'
    cmd += ' --Observations_coadd 0'
    cmd += ' --radius 0.01'
    cmd += ' --Output_directory {}'.format(outDir_simu)
    cmd += ' --Simulator_name sn_simulator.{}'.format(simulator)
    cmd += ' --Multiprocessing_nproc 1'
    cmd += ' --RAmin 0.0'
    cmd += ' --RAmax 0.1'
    cmd += '  --ProductionID {}'.format(prodid)
    cmd += ' --SN_ebvofMW {}'.format(ebvofMW)
    cmd += ' --SN_blueCutoff {}'.format(bluecutoff)
    cmd += ' --SN_redCutoff {}'.format(redcutoff)
    cmd += ' --npixels -1'
    cmd += ' --Simulator_errorModel {}'.format(error_model)
    cmd += ' --SN_maxRFphase 60.'
    print(cmd)
    os.system(cmd)
