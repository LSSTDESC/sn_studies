from sn_fom.cosmo_fit import zcomp_pixels, FitCosmo
from sn_fom.utils import loadSN, selSN, update_config
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from . import np

def fit_SN(fileDir, dbNames, config, fields, saveSN=''):
    data_sn = pd.DataFrame()
    for i, dbName in enumerate(dbNames):
        fields_to_process = fields[i].split(',')
        idx = config['fieldName'].isin(fields_to_process)
        dd = getSN(fileDir, dbName, config[idx], fields_to_process)
        data_sn = pd.concat((data_sn, dd))
        print('SN inter', dbName, len(dd))

    SNID = ''
    if saveSN != '':
        SNID = saveSN.split('.hdf5')[0]
        data_sn.to_hdf(saveSN, key='sn')
    # FitCosmo instance
    fit = FitCosmo(data_sn)

    # make the fit and get the parameters
    params_fit = fit()
    if SNID != '':
        params_fit['SNID'] = SNID
    return params_fit


def multifit(index, params, j=0, output_q=None):

    fileDir = params['fileDir']
    dbNames = params['dbNames']
    config = params['config']
    fields = params['fields']

    params_fit = pd.DataFrame()

    for i in index:
        saveSN = 'SN_{}.hdf5'.format(i)
        fitpar = fit_SN(fileDir, dbNames, config, fields, saveSN=saveSN)
        params_fit = pd.concat((params_fit, fitpar))

    if output_q is not None:
        return output_q.put({j: params_fit})
    else:
        return params_fit


def getSN(fileDir, dbName, config, fields):

    tt = zcomp_pixels(fileDir, dbName, 'faintSN')
    zcomp = tt()
    print('redshift completeness', zcomp)

    config = update_config(fields, config,
                           np.round(zcomp['zcomp'][0], 2))
    print('config updated', config)

    # get number of supernovae
    nsn_scen = NSN_config(config)
    print('total number of SN', nsn_scen.nsn_tot())
    nsn_per_bin = nsn_bin(nsn_scen.data)
    print(nsn_per_bin, nsn_per_bin['nsn_survey'].sum())

    # get SN from simu
    data_sn = loadSN(fileDir, dbName, 'allSN', zcomp)

    # select according to nsn_per_bin
    data_sn = selSN(data_sn, nsn_per_bin)

    print(len(data_sn), type(data_sn))

    return data_sn
