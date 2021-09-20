import matplotlib.pyplot as plt
import numpy as np
# from . import np
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
from sn_fom.steps import multifit
from sn_fom.plot import plotStat, plotHubbleResiduals, binned_data, plotFitRes
from sn_fom.utils import getconfig
from sn_fom.cosmo_fit import Sigma_Fisher
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
parser.add_option("--nproc", type=int,
                  default=4,
                  help="number of procs for multiprocessing  [%default]")
parser.add_option("--nseasons", type=int,
                  default=2,
                  help="number of seasons per field  [%default]")
parser.add_option("--dirSN", type=str,
                  default='fake_SN',
                  help="location dir of simulated SN used to estimate cosmo parameters  [%default]")
parser.add_option("--dirFit", type=str,
                  default='fake_Fit',
                  help="location dir of fit cosmo parameters  [%default]")


opts, args = parser.parse_args()

fileDir = opts.fileDir
dbNames = opts.dbName.split('/')
fields = opts.fields.split('/')
nproc = opts.nproc
dirSN = opts.dirSN
dirFit = opts.dirFit
nseasons = opts.nseasons
print('hello dbNames', dbNames, fields)
tagName = ''
for ip, dd in enumerate(dbNames):
    tagName += '{}_{}'.format(dd, fields[ip].replace(',', '_'))
    if ip < len(dbNames)-1:
        tagName += '_'

tagName += '_{}'.format(nseasons)
dirSN = '{}_{}'.format(dirSN, tagName)
dirFit = '{}_{}'.format(dirFit, tagName)

for vv in [dirSN, dirFit]:
    if not os.path.isdir(vv):
        os.mkdir(vv)

fitparName = '{}/FitParams.hdf5'.format(dirFit)

if not os.path.isfile(fitparName):
    # get default configuration file
    config = getconfig(nseasons=nseasons)

    ffi = range(16)
    params = {}
    params['fileDir'] = fileDir
    params['dbNames'] = dbNames
    params['config'] = config
    params['fields'] = fields
    params['dirSN'] = dirSN
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
print(params_fit[['SNID', 'M']])
Om = 0.3
w0 = -1.0
wa = 0.0
alpha = 0.13
beta = 3.1
M = -19.045

params = dict(zip(['M', 'alpha', 'beta', 'Om', 'w0', 'wa'],
                  [M, alpha, beta, Om, w0, wa]))
# params=dict(zip(['Om','w0','wa'],[Om,w0,wa]))
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

    plotresi = plotHubbleResiduals(row, snName)
    # plotresi.plots()
    plotresi.plot_sn_vars()

    data = pd.read_hdf(snName)
    print('NSN', len(data), data.columns)
    data['sigma_mu'] = data['Cov_mbmb']+alpha**2*data['Cov_x1x1']+beta**2*data['Cov_colorcolor'] + \
        2*alpha*data['Cov_x1mb']-2.*beta*data['Cov_colormb'] - \
        2.*alpha*beta*data['Cov_x1color']
    data['sigma_mu'] = np.sqrt(data['sigma_mu'])
    data['mu'] = -M+data['mbfit']+alpha*data['x1_fit']-beta*data['color_fit']
    sig = Sigma_Fisher(data, params=params)
    sig()
    for pp in ['M', 'alpha', 'beta', 'Om', 'w0']:
        print(pp, np.sqrt(row['Cov_{}_{}'.format(pp, pp)]))
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
