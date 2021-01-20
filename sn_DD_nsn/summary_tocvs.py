import numpy as np
import pandas as pd
from optparse import OptionParser


parser = OptionParser()

parser.add_option("--dbName", type=str, default='descddf_v1.5_10yrs',
                  help="OS to consider[%default]")
parser.add_option("--fieldName", type=str, default='COSMOS',
                  help="field to consider[%default]")
parser.add_option("--nside", type=float, default=128,
                  help="healpix nside[%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--zmax", type=float, default=0.8,
                  help="max redshift for simulated data [%default]")
parser.add_option("--ebvofMW", type=float, default=0.0,
                  help="E(B-V) value[%default]")
parser.add_option("--bluecutoff", type=float, default=380.,
                  help="blue cutoff for SN spectrum[%default]")
parser.add_option("--redcutoff", type=float, default=800.,
                  help="red cutoff for SN spectrum[%default]")
parser.add_option("--simulator", type=str, default='sn_fast',
                  help=" simulator to use[%default]")
parser.add_option("--snrmin", type=float, default=1.,
                  help="SNR min for LC points (fit)[%default]")
parser.add_option("--error_model", type=str, default='1',
                  help="error model to consider[%default]")
parser.add_option("--fitter", type='str', default='sn_fast',
                  help="fitter [%default]")


opts, args = parser.parse_args()

prefix = 'DD_Summary'
nside = opts.nside
dbName = opts.dbName
fieldName = opts.fieldName

fName = '{}_{}_{}.npy'.format(prefix, dbName, nside)

tab = np.load(fName, allow_pickle=True)

idx = tab['fieldName'] == fieldName

sel = tab[idx]

print(sel.dtype)
x1 = opts.x1
color = opts.color
ebvofMW = opts.ebvofMW
snrmin = opts.snrmin
error_model = opts.error_model
bluecutoff = opts.bluecutoff
redcutoff = opts.redcutoff
simulator = opts.simulator
fitter = opts.fitter

bands = 'grizy'
ro = []
for val in sel:
    #r = ['{}_{}'.format(val['healpixID'], int(val['season']))]
    r = [val['healpixID']]
    r += [x1, color, ebvofMW, snrmin, error_model,
          bluecutoff, redcutoff, simulator, fitter]
    for b in bands:
        r += [int(val['N_{}'.format(b)])]
    for b in bands:
        if val['N_{}'.format(b)] == 0:
            m5v = 21.
        else:
            m5v = val['m5_{}'.format(b)]-1.25*np.log10(val['N_{}'.format(b)])
        r += [m5v]
    for b in bands:
        r += [val['cadence']]
    r += [int(val['season'])]
    ro.append(r)

print(ro)
names = ['tagprod', 'x1', 'color', 'ebvofMW', 'snrmin', 'error_model', 'bluecutoff', 'redcutoff', 'simulator', 'fitter', 'Ng', 'Nr',
         'Ni', 'Nz', 'Ny', 'm5_g', 'm5_r', 'm5_i', 'm5_z', 'm5_y', 'cadence_g', 'cadence_r', 'cadence_i', 'cadence_z', 'cadence_y', 'season']
res = np.rec.fromrecords(ro, names=names)

print(res)
pd.DataFrame(res[:40]).to_csv('{}_cadence_{}.csv'.format(
    opts.dbName, opts.fieldName), index=None)
