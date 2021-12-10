import numpy as np
from optparse import OptionParser
import pandas as pd


def transform_df(dirFile, fileName):

    fName = '{}/{}'.format(opts.dirFile, opts.fileName)

    tab = np.load(fName, allow_pickle=True)

    df = pd.DataFrame(tab)

    df = df.sort_values(by=['Ny', 'zcomp'])
    df = df.round({'Nvisits_g': 0, 'Nvisits_r': 0, 'Nvisits_i': 0,
                   'Nvisits_z': 0, 'Nvisits_y': 0})
    for b in 'grizy':
        tt = 'Nvisits_{}'.format(b)
        df[tt] = df[tt].astype(int)

    df['Nvisits_g'] = 2
    df['Nvisits_r'] = 9
    df['dbName'] = 'DD_'+df['zcomp']+'_Ny_'+df['Ny'].astype(str)
    df['bands'] = 'grizy'
    return df


def config_lowz():

    r = []
    r.append(('DD_0.50', 2, 4, 1, 1, 0, 'griz'))
    r.append(('DD_0.55', 2, 7, 1, 1, 0, 'griz'))
    r.append(('DD_0.60', 2, 9, 1, 1, 0, 'griz'))
    r.append(('DD_0.65', 2, 9, 7, 9, 9, 'grizy'))

    columns = ['dbName', 'Nvisits_g', 'Nvisits_r',
               'Nvisits_i', 'Nvisits_z', 'Nvisits_y', 'bands']

    df = pd.DataFrame(r, columns=columns)

    print('ggg', df)
    return df


def combine_df(df, df_lowz):

    cols = ['dbName', 'Nvisits_g', 'Nvisits_r',
            'Nvisits_i', 'Nvisits_z', 'Nvisits_y', 'bands']

    for Ny in df['Ny'].unique():
        idx = df['Ny'] == Ny
        sel = df[idx]
        lowcp = df_lowz.copy()
        lowcp['dbName'] += '_Ny_{}'.format(Ny)
        sel = pd.concat((sel, lowcp))
        sel = sel.sort_values(by=['dbName'])
        print(sel[cols])
        outName = 'confignew_fakes_Ny_{}.csv'.format(Ny)
        sel[cols].to_csv(outName, index=False)


parser = OptionParser()

parser.add_option('--dirFile', type=str, default='dd_design/SNR_opti',
                  help='file location dir [%default]')
parser.add_option('--fileName', type=str, default='opti_-2.0_0.2_error_model_ebvofMW_0.0_cad_1.npy',
                  help='file to process[%default]')

opts, args = parser.parse_args()

df_lowz = config_lowz()
df = transform_df(opts.dirFile, opts.fileName)

combine_df(df, df_lowz)
