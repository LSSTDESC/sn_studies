import os


dbNames=['baseline_v1.5_10yrs',
         'descddf_v1.5_10yrs',
         'agnddf_v1.5_10yrs',
         'daily_ddf_v1.5_10yrs',
         'dm_heavy_nexp2_v1.6_10yrs',
         'dm_heavy_v1.6_10yrs',
         'ddf_heavy_nexp2_v1.6_10yrs',
         'ddf_heavy_v1.6_10yrs']

dbDirs = ['/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.5/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.5/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.5/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.5/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.6/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.6/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.6/npy',
          '/sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.6/npy']

toprocess = dict(zip(dbNames,dbDirs))

nside = 128
for key, val in toprocess.items():
    cmd = 'python for_batch/scripts/nsn_metric_DD.py'
    cmd += ' --pixelmap_dir /sps/lsst/users/gris/ObsPixelized_{}'.format(nside) 
    cmd += ' --outDir /sps/lsst/users/gris/MetricOutput_DD_new_{}'.format(nside) 
    cmd += ' --dbDir {}'.format(val) 
    cmd += ' --dbName {}'.format(key)
    cmd += ' --nside {}'.format(nside)
    print(cmd)
    os.system(cmd)
