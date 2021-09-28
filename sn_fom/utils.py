import numpy as np
import glob
from sn_tools.sn_utils import multiproc, x1_color_dist, check_get_file
from sn_tools.sn_io import loopStack_params
from sn_tools.sn_io import check_get_file
import pandas as pd


def update_config(fields, config, zcomp):

    idx = config['fieldName'].isin(fields)
    config.loc[idx, 'zcomp'] = zcomp

    return config


def loadSN(fDir, dbName, tagprod, zlim):

    # load data
    data = loadData(fDir, dbName, tagprod)

    # select data according to (zlim, season)

    if zlim > 0:
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


def selSN(sn_data, nsn_per_bin, x1_color):
    """
    Method to select a number of simulated SN according to the expected nsn_per_bin

    Parameters
    ---------------
    sn_data: astropy table
      simulated sn
    nsn_per_bin: pandas df
      df with the expected number of sn per bin
    x1_color: dict
      x1_color distribution for random choice

    Returns
    -----------
    selected data
    """

    surveytype = np.unique(nsn_per_bin['surveytype']).item()
    zcomp = np.unique(nsn_per_bin['zcomp']).item()
    zsurvey = np.unique(nsn_per_bin['zsurvey']).item()

    assert(surveytype in ['full', 'complete'])
    if surveytype == 'full':
        zmax = zsurvey
    if surveytype == 'complete':
        zmax = zcomp

    zmax = np.round(zmax, 2)
    zstep = 0.05
    zvals = np.arange(0, zmax+zstep, 0.05).tolist()

    zvals[0] = 0.01
    zvals[-1] = zmax

    #sel_data = select(sn_data)
    out_data = pd.DataFrame()
    for i in range(len(zvals)-1):
        zm = np.round(zvals[i], 2)
        zp = np.round(zvals[i+1], 2)
        ida = sn_data['z'] >= zm
        ida &= sn_data['z'] < zp
        sn_data_z = sn_data[ida]
        """
        idc = sel_data['z'] >= zm
        idc &= sel_data['z'] < zp
        """
        idb = np.abs(nsn_per_bin['z']-zp) < 1.e-5
        selz = nsn_per_bin[idb]
        nsn_expected = int(selz['nsn_survey'].values[0])
        #nsn_simu = len(sn_data[ida])
        #nsn_sel = len(sel_data[idc])
        #nsn_choose = int(nsn_sel/nsn_simu*nsn_expected)
        nsn_choose = nsn_expected
        #print('expected nsn',zp,nsn_expected)
        """
        if nsn_choose == 0:
            nsn_choose = 10
        print(zp, nsn_expected, nsn_simu, nsn_sel, nsn_choose)
        nsn_choose = nsn_sel
        """
        zrange = 'highz'
        if zp <= 0.1:
            zrange = 'lowz'
        if nsn_choose > 0:
            if zp <= 0.2:
                nsn_choose *= 20
            #x1_color_vals = pdist(sn_data, x1_color, zrange, nsn_choose)
            #print('my choice here',zm,zp,x1_color_vals)
            selected_data = pd.DataFrame(sn_data_z)
            selected_data['inum'] = selected_data.reset_index().index

            # select(x1,color) selected data according to dist

            #selected_data = select_x1_color(sn_data_z, x1_color_vals)

            choice = np.random.choice(len(selected_data), nsn_choose)
            #print('choice', choice, len(selected_data), nsn_choose)
            # print(selected_data)
            io = selected_data['inum'].isin(choice)
            out_data = pd.concat((out_data, selected_data[io]))

    # select sn here
    out_data = select(out_data)

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


def getconfig(fields=['COSMOS', 'XMM-LSS', 'ELAIS', 'CDFS', 'ADFS'], nseasons=2, max_season_length=180., survey_area=9.6, zsurvey=1., surveytype='full'):

    # fields = ['COSMOS']
    nfields = dict(zip(fields, [1, 1, 1, 1, 2]))
    zcomp = dict(zip(fields, [0.9, 0.9, 0.7, 0.7, 0.7]))

    # get scenario
    r = []

    for field in fields:
        r.append((field, zcomp[field], max_season_length,
                  nfields[field], survey_area, nseasons, zsurvey, surveytype))

    config = pd.DataFrame(r, columns=[
        'fieldName', 'zcomp', 'max_season_length', 'nfields', 'survey_area', 'nseasons', 'zsurvey', 'surveytype'])

    return config


def getDist(web_path='https://me.lsst.eu/gris/DESC_SN_pipeline',
            dirFiles='reference_files', distName='x1_color_G10.csv'):
    """ get (x1,color) distributions
    Parameters
    --------------
    web_path: str, opt
      web path where to find the file
      (default: https://me.lsst.eu/gris/DESC_SN_pipeline)
    dirFiles: str, opt
      dir where the file will be copied (frome web_path) (default: reference_files)
    distName: str, opt
       fileName to use (default: x1_color_G10.csv)

    Returns
    -----------
    pandas df with the following cols
    zrange, param, val, proba

    with
    zrange = lowz, highz
    param = x1,color
    """

    check_get_file(web_path, dirFiles, distName)

    fullName = '{}/{}'.format(dirFiles, distName)

    return x1_color_dist(fullName).proba


def pdist(df, x1_color, zrange, nsn_choose):

    idx = x1_color['zrange'] == zrange
    sel = x1_color[idx]

    # select x1 vals
    rr = {}
    for vv in ['x1', 'color']:
        ig = sel['param'] == vv
        selb = sel[ig]
        norm = np.sum(selb['proba'])
        rc = np.random.choice(df[vv], nsn_choose, p=selb['proba']/norm)
        rr[vv] = rc.tolist()

    df = pd.DataFrame.from_dict(rv)

    print(df)


def pdist_deprecated(x1_color, zrange, nsn_choose):

    df = pd.DataFrame(x1_color[zrange])
    df['inum'] = df.reset_index().index
    imin = df['inum'].min()
    imax = df['inum'].max()
    norm = np.sum(df['weight'])
    #print(imin, imax,len(df))
    ichoice = np.random.choice(
        range(imin, imax+1), nsn_choose, p=df['weight']/norm)

    return df.loc[ichoice][['x1', 'color']]


def select_x1_color(df, x1_color_vals):

    x1_color_vals = x1_color_vals.groupby(
        ['x1', 'color']).size().to_frame('size').reset_index()
    x1_color_vals['size'] = x1_color_vals['size'].astype(int)
    #print('for selection',x1_color_vals)
    new_df = pd.DataFrame()

    for i, vv in x1_color_vals.iterrows():
        nsn_th = int(vv['size'])
        idx = np.abs(df['x1']-vv['x1']) < 1.e-5
        idx &= np.abs(df['color']-vv['color']) < 1.e-5
        seldf = pd.DataFrame(df[idx])
        if len(seldf) > 0:
            seldf['inum'] = seldf.reset_index().index
            if len(seldf) <= vv['size']:  # take all here
                new_df = pd.concat((new_df, seldf))
            else:  # take random here
                #print('choice', len(seldf),nsn_th)
                choice = np.random.choice(len(seldf), nsn_th)
                #print('choice', choice,len(seldf), nsn_th)
                io = seldf['inum'].isin(choice)
                new_df = pd.concat((new_df, seldf[io]))

    return new_df


def transformSN(fileDir, dbName, snType, zcomp, alpha, beta):

    data_sn = loadSN(fileDir, dbName, snType, zcomp)
    data_sn = select(pd.DataFrame(data_sn))
    data_sn['Mb'] = -2.5*np.log10(data_sn['x0_fit'])+10.635
    data_sn['Cov_mbmb'] = (
        2.5 / (data_sn['x0_fit']*np.log(10)))**2*data_sn['Cov_x0x0']
    data_sn['Cov_x1mb'] = -2.5*data_sn['Cov_x0x1'] / \
        (data_sn['x0_fit']*np.log(10))

    data_sn['Cov_colormb'] = -2.5*data_sn['Cov_x0color'] / \
        (data_sn['x0_fit']*np.log(10))
    data_sn['var_mu'] = data_sn['Cov_mbmb']+alpha**2*data_sn['Cov_x1x1']+beta**2*data_sn['Cov_colorcolor'] + \
        2*alpha*data_sn['Cov_x1mb']-2.*beta*data_sn['Cov_colormb'] - \
        2.*alpha*beta*data_sn['Cov_x1color']
    data_sn['sigma_mu'] = np.sqrt(data_sn['var_mu'])

    return data_sn


def binned_data(zmin, zmax, nbins, data, var='sigma_mu'):
    """
    function  to transform a set of data to binned data

    Parameters
    ---------------
    zmin: float
      min redshift
    zmax: float
      max redshift
    data: pandas df
      data to be binned
    vary: str, opt
      y-axis variable (default: mu)
    erry: str, opt
      y-axis var error (default: sigma_mu)

    Returns
    -----------
    x, y, yerr:
    x : redshift centers
    y: weighted mean of distance modulus
    yerr: distance modulus error

    """
    bins = np.linspace(zmin, zmax, nbins)
    group = data.groupby(pd.cut(data.z, bins))
    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group[var].mean().to_list()
    error_values = group[var].std().to_list()

    print(plot_centers, plot_values)
    df = pd.DataFrame(plot_centers, columns=['z'])
    df['{}_mean'.format(var)] = plot_values
    df['{}_rms'.format(var)] = error_values
    return df
