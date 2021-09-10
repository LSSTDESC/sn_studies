import numpy as np
import glob
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import loopStack_params
import pandas as pd


def update_config(fields, config, zcomp):

    idx = config['fieldName'].isin(fields)
    config.loc[idx, 'zcomp'] = zcomp

    return config


def loadSN(fDir, dbName, tagprod, zlim):

    # load data
    data = loadData(fDir, dbName, tagprod)

    # select data according to (zlim, season)

    data = data.merge(zlim, left_on=['healpixID', 'season'], right_on=[
        'healpixID', 'season'])

    """
    data = select(data)
    idx = data['z']-data['zcomp'] <= 0.
    # idx &= data['z'] >= 0.1
    # idx = np.abs(data['z']-data['zcomp']) >= 0.
    data = data[idx].copy()

    # data = data[:100]
    """
    print('Number of SN here', len(data))

    return data


def selSN(sn_data, nsn_per_bin):
    """
    Method to select a number of simulated SN according to the expected nsn_per_bin

    Parameters
    ---------------
    sn_data: astropy table
      simulated sn
    nsn_per_bin: pandas df
      df with the expected number of sn per bin

    Returns
    -----------
    selected data
    """

    zcomp = np.unique(sn_data['zcomp']).item()
    zcomp = np.round(zcomp, 2)
    zstep = 0.05
    zvals = np.arange(0, zcomp+zstep, 0.05).tolist()

    zvals[0] = 0.01
    zvals[-1] = zcomp

    sel_data = select(sn_data)
    out_data = pd.DataFrame()
    for i in range(len(zvals)-1):
        zm = np.round(zvals[i], 2)
        zp = np.round(zvals[i+1], 2)
        ida = sn_data['z'] >= zm
        ida &= sn_data['z'] < zp
        idc = sel_data['z'] >= zm
        idc &= sel_data['z'] < zp
        idb = np.abs(nsn_per_bin['z']-zp) < 1.e-5
        selz = nsn_per_bin[idb]
        nsn_expected = int(selz['nsn_survey'].values[0])
        nsn_simu = len(sn_data[ida])
        nsn_sel = len(sel_data[idc])
        nsn_choose = int(nsn_sel/nsn_simu*nsn_expected)
        """
        if nsn_choose == 0:
            nsn_choose = 10
        print(zp, nsn_expected, nsn_simu, nsn_sel, nsn_choose)
        nsn_choose = nsn_sel
        """
        if nsn_choose > 0:
            selected_data = pd.DataFrame(sel_data[idc])
            selected_data['inum'] = selected_data.reset_index().index
            choice = np.random.choice(len(selected_data), nsn_choose)
            print('choice', choice, len(selected_data), nsn_choose)
            print(selected_data)
            io = selected_data['inum'].isin(choice)
            out_data = pd.concat((out_data, selected_data[io]))

    return out_data


def loadData(dirFile, dbName, tagprod):
    """
    Function to load data from file

    Parameters
    ---------------
    dirFile: str
      file directory
    dbName: str
       OS name
    tagprod: str
        tag for the file to load

    Returns
    -----------
    astropytable of data

    """
    search_path = '{}/{}/*{}*.hdf5'.format(dirFile, dbName, tagprod)
    print('search path', search_path)
    fis = glob.glob(search_path)

    print(fis)
    # load the files
    params = dict(zip(['objtype'], ['astropyTable']))

    res = multiproc(fis, params, loopStack_params, 4).to_pandas()
    # res['fitstatus'] = res['fitstatus'].str.decode('utf-8')

    return res


def select(dd):
    """"
    Function to select data

    Parameters
    ---------------
    dd: pandas df
      data to select

    Returns
    -----------
    selected pandas df

    """
    idx = dd['z'] < 1.2
    idx &= dd['z'] >= 0.01
    idx &= dd['fitstatus'] == 'fitok'
    idx &= np.sqrt(dd['Cov_colorcolor']) <= 0.04

    return dd[idx].copy()


def getconfig(fields=['COSMOS', 'XMM-LSS', 'ELAIS', 'CDFS', 'ADFS'], nseasons=2, max_season_length=180., survey_area=9.6):

    # fields = ['COSMOS']
    nfields = dict(zip(fields, [1, 1, 1, 1, 2]))
    zcomp = dict(zip(fields, [0.9, 0.9, 0.7, 0.7, 0.7]))

    # get scenario
    r = []

    for field in fields:
        r.append((field, zcomp[field], max_season_length,
                  nfields[field], survey_area, nseasons))

    config = pd.DataFrame(r, columns=[
        'fieldName', 'zcomp', 'max_season_length', 'nfields', 'survey_area', 'nseasons'])

    return config
