from sn_fom.utils import loadSN, binned_data
import matplotlib.pyplot as plt
import numpy as np


def loadData(fDir, dbName):

    data = loadSN(fDir, dbName, 'mediumSN')

    idx = data['fitstatus'] == 'fitok'
    data = data[idx]
    data['sigmaC'] = np.sqrt(data['Cov_colorcolor'])

    return data


fDir = 'Fakes_medium/Fit'
dbNames = ['DD_0.70_Ny_10', 'DD_0.70_Ny_20']

data = {}

for dbName in dbNames:
    data[dbName] = loadData(fDir, dbName)

fig, ax = plt.subplots()

zmin = 0.01
zmax = 1.10
nbins = 60

for key, vals in data.items():
    bdata = binned_data(zmin, zmax, nbins, vals, 'sigmaC')
    ax.plot(bdata['z'], bdata['sigmaC_mean'], label=key)

ax.grid()
ax.legend()
ax.set_ylim([0.0, 0.05])
plt.show()
