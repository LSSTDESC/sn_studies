import matplotlib.pyplot as plt
import numpy as np
# from . import np
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
from sn_fom.steps import multifit_mu, Sigma_mu_obs, SN_WFD, NSN_bias
from sn_fom.plot import plotStat, plotHubbleResiduals, binned_data, plotFitRes, plotSN
from sn_fom.plot import plotHubbleResiduals_mu
from sn_fom.utils import getconfig
from sn_fom.cosmo_fit import Sigma_Fisher, Sigma_Fisher_mu
import os
import pandas as pd


def go_fit(nMC, params, nproc, fitparName):

    ffi = range(nMC)
    params_fit = multiproc(ffi, params, multifit_mu, nproc)

    params_fit.to_hdf(fitparName, key='fitparams')


def plotFitResults(params_fit):

    plotFitRes(params_fit)
    plt.show()
    fig, ax = plt.subplots()

    for i, row in params_fit.iterrows():
        snName = '{}.hdf5'.format(row['SNID'])
        print('snname', snName)
        plotresi = plotHubbleResiduals_mu(
            row, snName, var_FoM=['Om', 'w0'], var_fit=parameter_to_fit)
        plotresi.plots()
        plt.show()


def Fisher_mu(params_fit, params):

    for i, row in params_fit.iterrows():
        snName = '{}.hdf5'.format(row['SNID'])
        data = pd.read_hdf(snName)
        sig = Sigma_Fisher_mu(
            data, params=params, params_Fisher=parameter_to_fit)
        res_Fisher = sig()
        for pp in sig.params_Fisher:
            row['sigma_{}'.format(pp)] = np.sqrt(
                row['Cov_{}_{}'.format(pp, pp)])
            print(pp, row[pp], row['sigma_{}'.format(pp)],
                  res_Fisher[pp], row['sigma_{}'.format(pp)]/res_Fisher[pp])
        print('chi2', row['chi2'])


def prepareOut(dirSN, dirFit, dbNames, fields, add_WFD, tagIt=False):
    """
    Method to prepare output dir

    Parameters
    --------------
    dirSN: str
      location dir of the simulated SN
    dirFit: str
      location dir of fit values
    dbNames: list(str)
      list of DbNames to process
    fields: list(str)
      list of fields to consider
    add_WFD: str
        WFD file name
    tagIt: bool, opt
      to tag the output dir or not (default: False)

    Returns
    ----------
    dirSN: str
      location dir of the simulated SN
    dirFit: str
      location dir of fit values

    """
    if tagIt:
        tagName = ''
        for ip, dd in enumerate(dbNames):
            tagName += '{}_{}'.format(dd, fields[ip].replace(',', '_'))
            if ip < len(dbNames)-1:
                tagName += '_'

        for seas in nseasons:
            ns = seas.split(',')
            tagName += '_'
            tagName += '_'.join(ns)

        tagName += '_{}'.format(snType)

        if add_WFD != '':
            tagName += '_{}'.format(add_WFD)

        dirSN = '{}_{}'.format(dirSN, tagName)
        dirFit = '{}_{}'.format(dirFit, tagName)

    for vv in [dirSN, dirFit]:
        if vv != '':
            if not os.path.isdir(vv):
                os.makedirs(vv)

    return dirSN, dirFit


parser = OptionParser(
    description='perform cosmo fit')
parser.add_option("--fileDir", type="str",
                  default='Fakes_nosigmaInt/Fit',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")
parser.add_option("--fields", type="str",
                  default='COSMOS, XMM-LSS, ELAIS, CDFS, ADFS',
                  help="file directory [%default]")
parser.add_option("--add_WFD", type=str,
                  default='',
                  help="name of the WFD SN file to add [%default]")
parser.add_option("--nproc", type=int,
                  default=4,
                  help="number of procs for multiprocessing  [%default]")
parser.add_option("--nseasons", type=str,
                  default='2,2,2,2,2',
                  help="number of seasons per field  - DDFs [%default]")
parser.add_option("--npointings", type=str,
                  default='1,1,1,1,2',
                  help="number of pointings per field  - DDFs [%default]")
parser.add_option("--dirSN", type=str,
                  default='fake_SN',
                  help="location dir of simulated SN used to estimate cosmo parameters  [%default]")
parser.add_option("--dirFit", type=str,
                  default='fake_Fit',
                  help="location dir of fit cosmo parameters  [%default]")
parser.add_option("--snType", type=str,
                  default='allSN',
                  help="SN type for main run (faintSN,mediumSN,brightSN,allSN)  [%default]")
parser.add_option("--surveyType", type=str,
                  default='full',
                  help="type of survey (full, complete) [%default]")
parser.add_option("--zsurvey", type=float,
                  default=1.2,
                  help="zmax for the survey [%default]")
parser.add_option("--nMC", type=int,
                  default=100,
                  help="number of cosmo trials [%default]")
parser.add_option("--sigmaInt", type=float,
                  default=0.12,
                  help="sigmaInt for cosmo MC [%default]")
parser.add_option("--configName", type=str,
                  default='config1',
                  help="configName for output [%default]")
parser.add_option("--binned_cosmology", type=int,
                  default=0,
                  help="to perform a binned cosmology  [%default]")
parser.add_option("--dbNames_all", type=str,
                  default='DD_0.50,DD_0.55,DD_0.60,DD_0.65,DD_0.70,DD_0.75,DD_0.80,DD_0.85,DD_0.90',
                  help="dbNames to consider to estimate reference files [%default]")
parser.add_option("--fit_parameters", type=str, default='Om,w0,wa',
                  help="parameters to fit [%default]")
parser.add_option("--Ny", type=int, default=40,
                  help="y-band visits max at 0.9 [%default]")
parser.add_option("--sigma_mu_photoz", type=str, default='',
                  help="mu error from photoz [%default]")
parser.add_option("--sigma_mu_bias_x1_color", type=str, default='sigma_mu_bias_x1_color_1_sigma',
                  help="mu error bias from x1 and color n-sigma variation [%default]")
parser.add_option("--sigma_mu_simu", type=str, default='sigma_mu_from_simu_Ny_40',
                  help="sigma_mu file for distance moduli simulation [%default]")
parser.add_option("--nsn_bias_simu", type=str, default='nsn_bias_Ny_40',
                  help="nsn_bias file for distance moduli simulation [%default]")

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbNames = opts.dbName.split('/')
fields = opts.fields.split('/')
nproc = opts.nproc
dirSN = opts.dirSN
dirFit = opts.dirFit
nseasons = opts.nseasons.split('/')
npointings = opts.npointings.split('/')
snType = opts.snType
surveyType = opts.surveyType
zsurvey = opts.zsurvey
add_WFD = opts.add_WFD
nMC = opts.nMC
sigmaInt = opts.sigmaInt
configName = opts.configName
binned_cosmology = opts.binned_cosmology
dbNames_all = opts.dbNames_all .split(',')
Ny = opts.Ny
sigma_mu_photoz = opts.sigma_mu_photoz
sigma_mu_bias_x1_color = opts.sigma_mu_bias_x1_color
sigma_mu_simu = opts.sigma_mu_simu
nsn_bias_simu = opts.nsn_bias_simu

"""
dbC = []
for dbN in dbNames_all:
    dbC.append(dbN+'_Ny_{}'.format(Ny))

dbNames_all = dbC
"""
# load sigma_mu
outName = '{}.hdf5'.format(sigma_mu_simu)
sigma_mu_from_simu = Sigma_mu_obs(fileDir,
                                  dbNames=dbNames_all+['WFD_0.20'],
                                  snTypes=['allSN']*len(dbNames_all)+['WFD'],
                                  outName=outName, plot=False).data
# print('boo', sigma_mu_from_simu)
# print(test)
# load nsn_bias
# special config file needed here: 1 season, 1 pointing per field
config = getconfig(['DD_0.90'],
                   ['COSMOS,XMM-LSS,CDFS,ADFS,ELAIS'],
                   ['1,1,1,1,1'],
                   ['1,1,1,1,1'])

outName = '{}.hdf5'.format(nsn_bias_simu)
nsn_bias = NSN_bias(fileDir, config, fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                    dbNames=dbNames_all,
                    plot=False, outName=outName).data

"""

idx = nsn_bias['zcomp'] == '0.80'
idx &= nsn_bias['fieldName'] == 'XMM-LSS'
idx &= nsn_bias['z'] <= 1.1
sel = nsn_bias[idx]
print('number of SN', np.sum(sel['nsn_eff']))
"""

# load sn_wfd
sn_wfd = pd.DataFrame()
if add_WFD != '':
    nsn_mu_simu = int(add_WFD.split('_')[2].split('.')[0])
    sn_wfd = SN_WFD(fileDir, sigma_mu_from_simu,
                    saveSN='{}.hdf5'.format(add_WFD), nfich=-1, nsn=nsn_mu_simu)

dirSN, dirFit = prepareOut(dirSN, dirFit, dbNames, fields, add_WFD)


# get default config file here
config = getconfig(dbNames, fields, nseasons, npointings, zsurvey=zsurvey,
                   surveytype=surveyType)


fitparName = '{}/FitParams_{}.hdf5'.format(dirFit, configName)
parameter_to_fit = opts.fit_parameters.split(',')

params = {}
params['fileDir'] = fileDir
params['dbNames'] = dbNames
params['config'] = config
params['fields'] = fields
params['dirSN'] = ''
params['snType'] = snType
params['sigma_mu'] = sigma_mu_from_simu
params['params_fit'] = parameter_to_fit
params['nsn_bias'] = nsn_bias
params['sn_wfd'] = sn_wfd
params['sigma_bias_x1_color'] = pd.read_hdf(
    '{}.hdf5'.format(sigma_mu_bias_x1_color))
params['sigmaInt'] = sigmaInt
params['binned_cosmology'] = binned_cosmology
params['surveyType'] = surveyType
params['sigma_mu_photoz'] = pd.DataFrame()
if sigma_mu_photoz != '':
    params['sigma_mu_photoz'] = pd.read_hdf('{}.hdf5'.format(sigma_mu_photoz))

go_fit(nMC, params, nproc, fitparName)

params_fit = pd.read_hdf(fitparName)
idx = params_fit['accuracy'] == 1
params_fit = params_fit[idx]
print('result', np.median(params_fit['sigma_w0']),
      np.std(params_fit['sigma_w0']), np.median(params_fit['nsn_DD']), np.median(params_fit['nsn_DD_COSMOS']+params_fit['nsn_DD_XMM-LSS']), np.median(params_fit['nsn_DD_XMM-LSS']))

# plotFitResults(params_fit)
"""
Om = 0.3
w0 = -1.0
wa = 0.0
params = dict(zip(parameter_to_fit, [Om, w0, wa]))
Fisher_mu(params_fit, params)
"""
