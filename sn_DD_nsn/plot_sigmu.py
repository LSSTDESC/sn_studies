from optparse import OptionParser
import h5py
from sn_tools.sn_io import loopStack
import matplotlib.pyplot as plt
import numpy as np


def loadFile(fileDir, fileName):
    fullName = '{}/{}'.format(fileDir, fileName)

    sn = loopStack([fullName], 'astropyTable')

    return sn


def plot(ax, sn):

    sn.sort('z')
    ax.plot(sn['z'], sn['sigma_mu'], color='r')
    ax.plot(sn['z'], sn['beta']*np.sqrt(sn['Cov_colorcolor']), color='k')
    ax.plot(sn['z'], sn['alpha']*np.sqrt(sn['Cov_x1x1']), color='b')
    ax.plot(sn['z'], np.sqrt(sn['Cov_mbmb']), color='g')


parser = OptionParser()

parser.add_option(
    '--fileDir', help='file directory [%default]', default='zlim_test/Output_Fit_error_model_ebvofMW_0.0_snrmin_1_errmodrel_0.1', type=str)
parser.add_option(
    '--fileName', help='file to process [%default]', default='Fit_sn_cosmo_Fake_-2.0_0.2_error_model_ebvofMW_0.0_0_sn_cosmo.hdf5', type=str)
parser.add_option(
    '--fileNameb', help='2nd file to process [%default]', default='Fit_sn_fast_Fake_-2.0_0.2_error_model_ebvofMW_0.0_1_sn_fast.hdf5', type=str)


opts, args = parser.parse_args()

fileDir = opts.fileDir
fileName = opts.fileName
fileNameb = opts.fileNameb

sn = loadFile(fileDir, fileName)
snb = loadFile(fileDir, fileNameb)

fig, ax = plt.subplots()

plot(ax, sn)
#plot(ax, snb)
print(sn.columns)

plt.show()
