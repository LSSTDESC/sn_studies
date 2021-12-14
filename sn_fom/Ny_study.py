import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sn_tools.sn_rate import SN_Rate
from sn_fom import plt
from scipy.interpolate import make_interp_spline, interp1d
from scipy.ndimage.filters import gaussian_filter1d


def load_files(Ny=[20, 30, 40, 60, 80, 120], prefix='nsn_bias_Ny', preconfig='config_fakes_Ny'):
    df = {}
    config = {}
    for vv in Ny:
        df[vv] = pd.read_hdf('{}_{}.hdf5'.format(prefix, vv))
        config[vv] = pd.read_csv('{}_{}.csv'.format(preconfig, vv))

    return df, config


def select_df(df, zcomp, fieldName, zcol='zcomp'):
    sel = {}
    for key, vals in df.items():
        print(vals.columns)
        sel[key] = select(vals, zcomp, fieldName, zcol)
    return sel


def nsn(sel, zcut):
    for key, vals in sel.items():
        idx = vals['z'] >= zcut
        print(key, np.sum(vals[idx]['nsn_eff']))


def select(data, zcomp, fieldName, zcol='zcomp'):

    idx = data[zcol] == zcomp
    if fieldName != '':
        idx &= data['fieldName'] == fieldName
    idx &= data['z'] <= 1.1
    return data[idx]


def get_infos(df, config, fieldName='XMM-LSS'):

    Nys = [20, 30, 40, 60, 80, 120]
    zcomps = ['0.70', '0.75', '0.80', '0.85', '0.90']

    r = []
    for Ny in Nys:
        for zcomp in zcomps:
            idx = df[Ny]['zcomp'] == zcomp
            idx &= df[Ny]['fieldName'] == fieldName
            idxb = config[Ny]['dbName'] == 'DD_{}_Ny_{}'.format(zcomp, Ny)

            seldf = df[Ny][idx]
            idxc = seldf['z'] >= 0.9
            idxc &= seldf['z'] <= 1.1
            selldf = seldf[idxc]
            conf = config[Ny][idxb]
            Nvisits = 0
            Nvisits_y = conf['Nvisits_y'].to_list()[0]
            for b in 'grizy':
                Nvisits += conf['Nvisits_{}'.format(b)].to_list()[0]
            print(Ny, zcomp, np.sum(selldf['nsn_eff']), Nvisits, Nvisits_y)
            r.append(
                (Ny, zcomp, np.sum(selldf['nsn_eff']), Nvisits, Nvisits_y))

    outdf = pd.DataFrame(
        r, columns=['Ny', 'zcomp', 'NSN_0.90', 'Nvisits', 'Nvisits_y'])
    return outdf


def get_expected_SN(H0=70., Om0=0.3, min_rf_phase=-10., max_rf_phase=25., zmin=0.01, zmax=1.1, dz=0.01, season_length=180., survey_area=9.6, account_for_edges=True):

    rateSN = SN_Rate(H0=H0, Om0=Om0,
                     min_rf_phase=min_rf_phase, max_rf_phase=max_rf_phase)

    zz, rate, err_rate, nsn, err_nsn = rateSN(zmin=zmin,
                                              zmax=zmax,
                                              dz=dz,
                                              duration=season_length,
                                              survey_area=survey_area,
                                              account_for_edges=account_for_edges)

    nsn_tot = np.cumsum(nsn)[-1]
    idx = zz >= 0.9
    nsn_z = np.cumsum(nsn[idx])[-1]

    return nsn_tot, nsn_z


def plot(ax, res, zcomp, norm=1, xvar='Nvisits_y', yvar='Nvisits'):

    if isinstance(norm, pd.DataFrame):
        io = refdf['zcomp'] == zcomp
        norm = refdf[io][yvar].to_list()[0]

    idx = res['zcomp'] == zcomp
    seldf = res[idx]
    seldf = seldf.sort_values(by=['Nvisits_y'])
    print(zcomp, seldf)
    seldf = seldf.groupby([xvar])[yvar].mean().reset_index()
    y = seldf[yvar]/norm
    x = seldf[xvar]
    ax.plot(seldf[xvar], y,
            label='$z_{complete}$'+'= {}'.format(zcomp), lw=3, marker='o')

    f_cubic = interp1d(x, y, kind='quadratic')
    xmin, xmax = np.min(x), np.max(x)
    xnew = np.linspace(xmin, xmax, 100)
    ax.plot(xnew, f_cubic(xnew),
            label='$z_{complete}$'+'= {}'.format(zcomp), lw=3)

    """
    xmin, xmax = np.min(seldf[xvar]), np.max(seldf[xvar])
    xnew = np.linspace(xmin, xmax, 100)
    spl = make_interp_spline(
        seldf[xvar], seldf[yvar]/norm, k=1)  # type: BSpline
    spl_smooth = spl(xnew)
    ax.plot(xnew, spl_smooth,
            label='$z_{complete}$'+'= {}'.format(zcomp), lw=3)
    """


df, config = load_files()

print(df)
res = get_infos(df, config, fieldName='CDFS')

print(res)

nsn_tot, nsn_z = get_expected_SN()
print(nsn_tot, nsn_z)
fig, ax = plt.subplots()
figb, axb = plt.subplots()
#axb = ax.twinx()
idd = np.abs(res['Ny']-20) < 1.e-5
refdf = res[idd]

print('hhh', res['zcomp'].unique())
for zcomp in res['zcomp'].unique():
    plot(ax, res, zcomp, norm=nsn_z, xvar='Nvisits_y', yvar='NSN_0.90')
    plot(axb, res, zcomp, norm=refdf)


ax.set_xlabel('$y$-band visits')
ax.set_ylabel('fraction of SN $z\geq$0.9')
ax.set_xlim([4., None])
ax.grid()

axb.set_xlabel('$y$-band visits')
axb.set_ylabel(
    'N$_{\mathrm{visits}}^{\mathrm{N}_y=\mathrm{N}_y^{min}}$/N$_{\mathrm{visits}}$')
axb.set_ylim([None, 1.])
axb.set_xlim([4., None])
axb.grid()

ax.legend()
axb.legend()
plt.show()

print(test)


zcomp = '0.80'
# zcomp = zcomp[::-1].zfill(4)[::-1]
fieldName = 'XMM-LSS'
"""
df = load_files()
sel = select_df(df, zcomp, fieldName)
nsn(sel, 0.9)

fig, ax = plt.subplots()

for key, vals in sel.items():
    ax.plot(vals['z'], vals['nsn_eff'])
"""
df, config = load_files(prefix='sigma_mu_from_simu_Ny')
sel = select_df(df, 'DD_{}'.format(zcomp), '', zcol='dbName')

fig, ax = plt.subplots()

for key, vals in sel.items():
    ax.plot(vals['z'], vals['sigma_mu_mean'])

fig, ax = plt.subplots()
for key, vals in sel.items():
    ax.plot(vals['z'], vals['sigma_mu_sigma'])

plt.show()
