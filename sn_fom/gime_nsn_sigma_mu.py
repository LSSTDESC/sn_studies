from optparse import OptionParser
from sn_fom.utils import getconfig
from sn_fom.steps import Sigma_mu_obs, NSN_bias

parser = OptionParser(
    description='perform cosmo fit')
parser.add_option("--fileDir", type="str",
                  default='Fakes_nosigmaInt/Fit',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="dbNames to process [%default]")
parser.add_option("--fields", type="str",
                  default='COSMOS, XMM-LSS, ELAIS, CDFS, ADFS',
                  help="fields to process[%default]")
parser.add_option("--outName_sigmamu", type="str",
                  default='sigma_mu_from_simu_Ny_40',
                  help="outName for sigma_mu [%default]")
parser.add_option("--outName_nsn", type="str",
                  default='nsn_bias_Ny_40',
                  help="outName for nsn_bias[%default]")
parser.add_option("--Ny", type=int, default=40,
                  help="y-band visits max at 0.9 [%default]")
parser.add_option("--dbNames_all", type=str,
                  default='DD_0.65,DD_0.70,DD_0.75,DD_0.80,DD_0.85,DD_0.90',
                  help="dbNames to consider to estimate reference files [%default]")

opts, args = parser.parse_args()

# load sigma_mu
#outName = 'sigma_mu_from_simu_Ny_{}.hdf5'.format(Ny)
fileDir = opts.fileDir
dbNames = opts.dbName.split('/')
dbNames_all = opts.dbNames_all .split(',')
outName = '{}.hdf5'.format(opts.outName_sigmamu)
sigma_mu_from_simu = Sigma_mu_obs(fileDir,
                                  dbNames=dbNames_all,
                                  snTypes=['allSN']*len(dbNames_all),
                                  outName=outName, plot=False).data
# print('boo', sigma_mu_from_simu)
# print(test)
# load nsn_bias
# special config file needed here: 1 season, 1 pointing per field
config = getconfig(['DD_0.90'],
                   ['COSMOS,XMM-LSS,CDFS,ADFS,ELAIS'],
                   ['1,1,1,1,1'],
                   ['1,1,1,1,1'])

#outName = 'nsn_bias_Ny_{}.hdf5'.format(Ny)
outName = '{}.hdf5'.format(opts.outName_nsn)
nsn_bias = NSN_bias(fileDir, config, fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                    dbNames=dbNames_all,
                    plot=False, outName=outName).data
