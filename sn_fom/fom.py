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


def prepareOut(dirSN, dirFit, dbNames, fields, add_WFD):
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

    Returns
    ----------
    dirSN: str
      location dir of the simulated SN
    dirFit: str
      location dir of fit values

    """
    tagName = ''
    for ip, dd in enumerate(dbNames):
        tagName += '{}_{}'.format(dd, fields[ip].replace(',', '_'))
        if ip < len(dbNames)-1:
            tagName += '_'

    #tagName += '_{}'.format(nseasons)
    #    print('ho', nseasons)
    #    print(test)
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
        if not os.path.isdir(vv):
            os.mkdir(vv)

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
                  default=1.0,
                  help="zmax for the survey [%default]")

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


# load sigma_mu
sigma_mu_from_simu = Sigma_mu_obs(fileDir,
                                  outName='sigma_mu_from_simu.hdf5',
                                  plot=False).data

# load nsn_bias
# special config file needed here: 1 season, 1 pointing per field
config = getconfig(['DD_0.90'],
                   ['COSMOS,XMM-LSS,CDFS,ADFS,ELAIS'],
                   ['1,1,1,1,1'],
                   ['1,1,1,1,1'])
nsn_bias = NSN_bias(fileDir, config, fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                    dbNames=['DD_0.65', 'DD_0.70', 'DD_0.75',
                             'DD_0.80', 'DD_0.85', 'DD_0.90'],
                    plot=False).data

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


fitparName = '{}/FitParams.hdf5'.format(dirFit)
parameter_to_fit = ['Om', 'w0', 'wa']
if not os.path.isfile(fitparName):
    # get default configuration file

    ffi = range(80)
    params = {}
    params['fileDir'] = fileDir
    params['dbNames'] = dbNames
    params['config'] = config
    params['fields'] = fields
    params['dirSN'] = dirSN
    params['snType'] = snType
    params['sigma_mu'] = sigma_mu_from_simu
    params['params_fit'] = parameter_to_fit
    params['nsn_bias'] = nsn_bias
    params['sn_wfd'] = sn_wfd
    params['sigma_bias'] = 0.0
    params['sigmaInt'] = 0.12

    params_fit = multiproc(ffi, params, multifit_mu, nproc)

    params_fit.to_hdf(fitparName, key='fitparams')

params_fit = pd.read_hdf(fitparName)
idx = params_fit['accuracy'] == 1
params_fit = params_fit[idx]
print('result', np.median(params_fit['sigma_w0']),
      np.std(params_fit['sigma_w0']))

plotFitResults(params_fit)
"""
Om = 0.3
w0 = -1.0
wa = 0.0
params = dict(zip(parameter_to_fit, [Om, w0, wa]))
Fisher_mu(params_fit, params)
"""
