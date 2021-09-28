from sn_fom.cosmo_fit import zcomp_pixels, FitData
from sn_fom.utils import loadSN, selSN, update_config, getDist, transformSN, binned_data
from sn_fom.nsn_scenario import NSN_config, nsn_bin
import pandas as pd
from . import np


def fit_SN(fileDir, dbNames, config, fields, snType, sigmu=pd.DataFrame(), saveSN=''):
    data_sn = pd.DataFrame()

    for i, dbName in enumerate(dbNames):
        fields_to_process = fields[i].split(',')
        idx = config['fieldName'].isin(fields_to_process)
        #dd = getSN(fileDir, dbName, config[idx], fields_to_process, snType)
        idxb = sigmu['dbName'] == dbName
        dd = getSN_mu_simu(dbName, config[idx], sigmu[idxb])
        data_sn = pd.concat((data_sn, dd))
        print('SN inter', dbName, len(dd))

    print(test)
    SNID = ''
    if saveSN != '':
        SNID = saveSN.split('.hdf5')[0]
        data_sn.to_hdf(saveSN, key='sn')
    # FitCosmo instance
    fit = FitData(data_sn)

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
    saveSN = params['dirSN']
    snType = params['snType']
    sigma_mu_from_simu = params['sigma_mu']
    params_fit = pd.DataFrame()
    np.random.seed(123456+j)
    for i in index:
        if saveSN != '':
            saveSN_f = '{}/SN_{}.hdf5'.format(saveSN, i)
        fitpar = fit_SN(fileDir, dbNames, config,
                        fields, snType, sigma_mu_from_simu, saveSN=saveSN_f)
        params_fit = pd.concat((params_fit, fitpar))

    if output_q is not None:
        return output_q.put({j: params_fit})
    else:
        return params_fit


def getSN(fileDir, dbName, config, fields, snType):

    tt = zcomp_pixels(fileDir, dbName, 'faintSN')
    zcomp = tt()
    print('redshift completeness', zcomp)
    zcomplete = zcomp['zcomp'][0]
    #zcomplete = 1.
    config = update_config(fields, config,
                           np.round(zcomplete, 2))
    print('config updated', config)

    # get number of supernovae
    nsn_scen = NSN_config(config)
    print('total number of SN', nsn_scen.nsn_tot())
    nsn_per_bin = nsn_bin(nsn_scen.data)
    print(nsn_per_bin, nsn_per_bin['nsn_survey'].sum())

    # get SN from simu
    data_sn = loadSN(fileDir, dbName, snType, zcomp)

    # load x1_c distrib
    x1_color = getDist()

    # select according to nsn_per_bin
    data_sn = selSN(data_sn, nsn_per_bin, x1_color)

    print(len(data_sn), type(data_sn))

    return data_sn


def getSN_mu_simu(dbName, config, sigmu_from_simu):

    nsn_scen = NSN_config(config)
    print('total number of SN', nsn_scen.nsn_tot())
    nsn_per_bin = nsn_bin(nsn_scen.data)
    print(nsn_per_bin, nsn_per_bin['nsn_survey'].sum())

    return nsn_scen


def sigma_mu(params, alpha=0.13, beta=3.1):
    """
    Function to estimate sigma_mu for SN

    Parameters
    ---------------
    params: dict
     parameters for the files to process
    alpha: float,opt
     standardisation parameter (alpha*x1 in mu)(default: 0.13)
    beta: float, opt
      standardisation parameter (beta*color in mu) (default: 3.1)

    """

    print(params)
    fileDir = params['fileDir']
    snType = params['snType']
    zcomp = -1.
    zmin = 0.0
    zmax = 1.0
    nbins = 21
    df = pd.DataFrame()
    for dbName in params['dbNames']:
        data_sn = transformSN(fileDir, dbName, snType, zcomp, alpha, beta)
        print('hello', data_sn)
        bdata = binned_data(zmin, zmax, nbins, data_sn)
        bdata['dbName'] = dbName
        df = pd.concat((df, bdata))

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for dbName in df['dbName'].unique():
        idx = df['dbName'] == dbName
        bdata = df[idx]
        ax.errorbar(bdata['z'], bdata['sigma_mu_mean'],
                    yerr=bdata['sigma_mu_rms'], label=dbName)

    ax.legend()
    plt.show()
    """
    return df
