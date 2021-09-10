import numpy as np
#from . import np
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
from sn_fom.steps import multifit
from sn_fom.plot import plotStat, plotHubbleResiduals
from sn_fom.utils import getconfig
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

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbNames = opts.dbName.split('/')
fields = opts.fields.split('/')
nproc = opts.nproc
print('hello dbNames', dbNames, fields)



fitparName = 'FitParams.hdf5'
if not os.path.isfile(fitparName):
    # get default configuration file
    config = getconfig()

    ffi = range(10)
    params = {}
    params['fileDir'] = fileDir
    params['dbNames'] = dbNames
    params['config'] = config
    params['fields'] = fields
    params_fit = multiproc(ffi, params, multifit, nproc)

    print(params_fit)

    params_fit.to_hdf(fitparName, key='fitparams')

params_fit = pd.read_hdf(fitparName)


"""
# plot FoMs here
plots = plotStat(params_fit)

plots.plotFoM()
"""
print(params_fit['SNID'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i,row in params_fit.iterrows():
    snName = '{}.hdf5'.format(row['SNID'])
    
    plotresi = plotHubbleResiduals(row,snName)
    plotresi.plots()
    
    data = pd.read_hdf(snName)
    print(data.columns)
    #ax.hist(np.sqrt(data['Cov_colorcolor']),histtype='step')
    ax.plot(data['z'], np.sqrt(data['Cov_colorcolor']),marker='.',lineStyle='None')
    plt.show()
    
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
