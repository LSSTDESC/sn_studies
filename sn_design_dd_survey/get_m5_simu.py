import numpy as np
import pandas as pd
from optparse import OptionParser
from sn_tools.sn_obs import season


def calc_season(grp):
    """
    Function to estimate the seasons and get the fieldname

    Parameters
    ---------------
    grp: pandas group
      data to process

    Returns
    -----------
    initial grp plus two columns: season and fieldname

    """
    names_simu = ['DD:COSMOS', 'DD:ECDFS', 'DD:EDFSa',
                  'DD:EDFSb', 'DD:ELAISS1', 'DD:XMM-LSS']
    field_name = ['COSMOS', 'ECDFS', 'ADFS1', 'ADFS2', 'ELAIS', 'XMM-LSS']
    corresp = dict(zip(names_simu, field_name))
    res = season(grp.to_records(index=False), mjdCol='mjd')

    df = pd.DataFrame(np.copy(res))
    df['fieldname'] = corresp[grp.name]

    return df


parser = OptionParser()

parser.add_option('--dirFiles', type=str, default='../../DB_Files',
                  help='dir location of the file [%default]')
parser.add_option('--dbName', type=str, default='descddf_v1.5_10yrs',
                  help='OS to process [%default]')

opts, args = parser.parse_args()

fullName = '{}/{}.npy'.format(opts.dirFiles, opts.dbName)

# get seasons
tab = np.load(fullName, allow_pickle=True)
print(tab.dtype)

df = pd.DataFrame(tab)

idx = df['note'].str.contains('DD')
print(np.unique(df[idx]['note']))

# get seasons

df_ddf = df[idx].groupby(['note']).apply(lambda x: calc_season(x))

print(df_ddf['note'].unique())

print('Median m5')
m5Col = 'fiveSigmaDepth'
grp = df_ddf.groupby(['band'])[m5Col].median().reset_index()

print(grp)

"""
grp = df_ddf.groupby(['fieldname', 'band'])[m5Col].median().reset_index()

print(grp)
"""
grp = df_ddf.groupby(['fieldname', 'band', 'season'])[
    m5Col].median().reset_index()

# print(grp)
outName = 'medValues_{}_DD.npy'.format(opts.dbName)
np.save(outName, grp.to_records(index=False))
