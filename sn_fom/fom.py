import matplotlib.pyplot as plt
import numpy as np
# from . import np
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
from sn_fom.steps import multifit, Sigma_mu_obs, SN_WFD, NSN_bias
from sn_fom.plot import plotStat, plotHubbleResiduals, binned_data, plotFitRes, plotSN
from sn_fom.plot import plotHubbleResiduals_mu
from sn_fom.utils import getconfig
from sn_fom.cosmo_fit import Sigma_Fisher, Sigma_Fisher_mu
import os
import pandas as pd

parser = OptionParser(
    description='Estimate zlim from simulation+fit data')
parser.add_option("--fileDir", type="str",
                  default='/sps/lsst/users/gris/DD/Fit',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs',
                  help="file directory [%default]")
parser.add_option("--fields", type="str",
                  default='COSMOS, XMM-LSS, ELAIS, CDFS, ADFS',
                  help="file directory [%default]")
parser.add_option("--add_WFD", type=int,
                  default=0,
                  help="to add WFD SN [%default]")
parser.add_option("--nproc", type=int,
                  default=4,
                  help="number of procs for multiprocessing  [%default]")
parser.add_option("--nseasons", type=int,
                  default=2,
                  help="number of seasons per field  - DDFs [%default]")
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
nseasons = opts.nseasons
snType = opts.snType
surveyType = opts.surveyType
zsurvey = opts.zsurvey
add_WFD = opts.add_WFD

# load sigma_mu
sigma_mu_from_simu = Sigma_mu_obs(fileDir, plot=False).data

# load nsn_bias
config = getconfig(nseasons=1, zsurvey=zsurvey,
                   surveytype=surveyType, nfields=[1, 1, 1, 1, 1])
nsn_bias = NSN_bias(fileDir, config, fields=['COSMOS', 'XMM-LSS', 'CDFS', 'ADFS', 'ELAIS'],
                    dbNames=['DD_0.65', 'DD_0.70', 'DD_0.80', 'DD_0.90'],
                    plot=False).data

# load sn_wfd
sn_wfd = pd.DataFrame()
if add_WFD:
    sn_wfd = SN_WFD(fileDir, sigma_mu_from_simu)

print('hello dbNames', dbNames, fields, nseasons)
tagName = ''
for ip, dd in enumerate(dbNames):
    tagName += '{}_{}'.format(dd, fields[ip].replace(',', '_'))
    if ip < len(dbNames)-1:
        tagName += '_'

tagName += '_{}'.format(nseasons)
tagName += '_{}'.format(snType)

if add_WFD:
    tagName += '_with_WFD'
else:
    tagName += '_no_WFD'

dirSN = '{}_{}'.format(dirSN, tagName)
dirFit = '{}_{}'.format(dirFit, tagName)

for vv in [dirSN, dirFit]:
    if not os.path.isdir(vv):
        os.mkdir(vv)

print('dirfit', dirFit)

fitparName = '{}/FitParams.hdf5'.format(dirFit)

print('hello', fitparName)
config = getconfig(nseasons=nseasons, zsurvey=zsurvey,
                   surveytype=surveyType)

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

    params_fit = multiproc(ffi, params, multifit, nproc)

    print(params_fit)

    params_fit.to_hdf(fitparName, key='fitparams')

params_fit = pd.read_hdf(fitparName)


"""
# plot FoMs here
plots = plotStat(params_fit)

plots.plotFoM()
"""
print(params_fit.columns)
Om = 0.3
w0 = -1.0
wa = 0.0
alpha = 0.13
beta = 3.1
M = -19.0906
#M = -18.97
params = dict(zip(['M', 'alpha', 'beta', 'Om', 'w0', 'wa'],
                  [M, alpha, beta, Om, w0, wa]))
params = dict(zip(['Om', 'w0', 'wa'],
                  [Om, w0, wa]))
# params=dict(zip(['Om','w0','wa'],[Om,w0,wa]))


#idx = params_fit['accuracy'] == 1
#params_fit = params_fit[idx]
print('params fit', len(params_fit))
"""
myplot = plotSN(params_fit, params)
myplot()
"""
plotFitRes(params_fit)
fig, ax = plt.subplots()
"""
figb, axb = plt.subplots()
for i,row in params_fit.iterrows():
    ax.plot(row['SNID'],np.sqrt(row['Cov_w0_w0']),'ko')
    axb.plot(row['SNID'],np.sqrt(row['Cov_wa_wa']),'ko')
plt.show()
"""
for i, row in params_fit.iterrows():
    snName = '{}.hdf5'.format(row['SNID'])
    print('snname', snName)
    plotresi = plotHubbleResiduals_mu(
        row, snName, var_FoM=['w0', 'wa'], var_fit=parameter_to_fit)
    plotresi.plots()
    # plotresi.plot_sn_vars()

    data = pd.read_hdf(snName)
    """
    print('NSN', len(data), data.columns)
    data['Mb'] = -2.5*np.log10(data['x0_fit'])+10.635
    data['Cov_mbmb'] = (
        2.5 / (data['x0_fit']*np.log(10)))**2*data['Cov_x0x0']

    data['Cov_x1mb'] = -2.5*data['Cov_x0x1'] / \
        (data['x0_fit']*np.log(10))

    data['Cov_colormb'] = -2.5*data['Cov_x0color'] / \
        (data['x0_fit']*np.log(10))
    data['sigma_mu'] = data['Cov_mbmb']+alpha**2*data['Cov_x1x1']+beta**2*data['Cov_colorcolor'] + \
        2*alpha*data['Cov_x1mb']-2.*beta*data['Cov_colormb'] - \
        2.*alpha*beta*data['Cov_x1color']
    data['sigma_mu'] = np.sqrt(data['sigma_mu'])
    data['mu'] = -M+data['mbfit']+alpha*data['x1_fit']-beta*data['color_fit']
    """
    sig = Sigma_Fisher_mu(data, params=params, params_Fisher=parameter_to_fit)
    res_Fisher = sig()
    for pp in sig.params_Fisher:
        row['sigma_{}'.format(pp)] = np.sqrt(row['Cov_{}_{}'.format(pp, pp)])
        print(pp, row[pp], row['sigma_{}'.format(pp)],
              res_Fisher[pp], row['sigma_{}'.format(pp)]/res_Fisher[pp])
    print('chi2', row['chi2'])
    # print(row)
    plt.show()
    """
    # ax.hist(np.sqrt(data['Cov_colorcolor']),histtype='step')
    plot_centers, plot_values, error_values = binned_data(
        0.005,0.905,data, 19,vary='mu',erry='')
    ax.plot(plot_centers, plot_values,marker='.',lineStyle='None')
    plt.show()
    """
print(test)


# plot the result
Om = params_fit['Om'].values[0]
w0 = params_fit['w0'].values[0]
wa = params_fit['wa'].values[0]
M = params_fit['M'].values[0]
alpha = params_fit['alpha'].values[0]
beta = params_fit['beta'].values[0]

fit.plot_hubble(0., Om, w0, wa, M, alpha, beta)

plt.show()


fom = FoM(fileDir, dbName, 'allSN', zcomp)
fom.plot_sn_vars()
# plt.show()

Om = 0.3
w0 = -1.0
wa = 0.0
alpha = 0.14
beta = 3.1
Mb = -19.0481
Mb = -19.039


h = 1.e-8
varFish = ['dOm', 'dw0', 'dwa', 'dalpha', 'dbeta', 'dMb']
parNames = ['Om', 'w0', 'wa', 'alpha', 'beta', 'Mb']
parameters = dict(zip(parNames, [Om, w0, wa, alpha, beta, Mb]))
epsilon = dict(zip(parNames, [0.]*len(parNames)))
for i in range(1):
    data = fom.data.sample(n=fom.NSN)
    fom.plot_data_cosmo(data, alpha=alpha, beta=beta,
                        Mb=Mb, binned=True, nbins=100)

    Fisher = np.zeros((len(varFish), len(varFish)))
    for vv in parNames:
        epsilon[vv] = h
        data['d{}'.format(vv)] = data.apply(
            lambda x: deriv(x, fom, parameters, epsilon)/2/h, axis=1, result_type='expand')
        epsilon[vv] = 0.0

    for i, vva in enumerate(varFish):
        for j, vvb in enumerate(varFish):
            Fisher[i, j] = np.sum((data[vva]*data[vvb])/data['sigma_mu']**2)
    print(Fisher)
    res = np.sqrt(np.diag(faster_inverse(Fisher)))
    print(res)
    print(len(data), 1./(res[0]*res[1]))
