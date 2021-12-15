import pandas as pd
from sn_fom import plt
import numpy as np


def plot(df, xvar='year', yvar='sigma_w'):

    fig, ax = plt.subplots(figsize=(15, 8))
    for key, vals in df.items():
        ax.plot(vals[xvar], vals[yvar], marker='o', label=key)

    ax.grid()
    ax.legend()


def load(fi):
    """
    Function to load file in pandas df

    Parameters
    ---------------
    fi: str
      name of the file (full path) to be loaded

    Returns
    ----------
    pandas df corresponding to the file
    """
    df = pd.read_hdf(fi)

    df['year'] = df['conf'].str.split('_').str.get(1).astype(int)
    df = df.sort_values(by=['year'])

    return df


def load_b(fi, zcomp=0.65):
    """
    Function to load file in pandas df

    Parameters
    ---------------
    fi: str
      name of the file (full path) to be loaded
    zcomp: float, opt
      redshift completeness (default: 0.65)

    Returns
    ----------
    pandas df corresponding to the file
    """
    df = pd.read_hdf(fi)

    df['zcomp_dd_unique'] = df.apply(lambda x: np.mean(x['zcomp_dd']), axis=1)
    idx = np.abs(df['zcomp_dd_unique']-zcomp) < 1.e-5

    sel = df[idx]
    sel = sel.sort_values(by=['sigma_w'], ascending=False)
    sel['year'] = sel.reset_index().index+1

    print(sel)

    return sel


df = {}
fi = 'cosmoSN_deep_rolling_2_2_mini_yearly_Ny_40.hdf5'

df['deep_rolling_early'] = load(fi)

fib = 'cosmoSN_deep_rolling_0.80_0.80_yearly_Ny_40.hdf5'

df['deep_rolling_ten_years'] = load(fib)

fic = 'cosmoSN_universal_yearly_Ny_40.hdf5'

df['universal_0.60'] = load_b(fic, zcomp=0.60)
df['universal_0.65'] = load_b(fic, zcomp=0.65)

plot(df)
plot(df, xvar='nsn_z_09')

plt.show()
