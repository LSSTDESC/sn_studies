import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'


def budget(Nfields, Nseasons, Nvisits, season_length, cadence):

    Nvisits_nonDD = 2122176
    Nvisits_DD = Nfields*Nseasons*Nvisits*season_length/cadence

    return Nvisits_DD/(Nvisits_DD+Nvisits_nonDD)


def budplot(Nfields, Nseasons, cadence, nvmin, nvmax, slmin, slmax):

    #nvmin, nvmax = 10,300
    #slmin, slmax = 100,250

    nvisits = np.linspace(nvmin, nvmax, 100)
    seaslength = np.linspace(slmin, slmax, 100)

    NV, SL = np.meshgrid(nvisits, seaslength)
    BUD = budget(Nfields, Nseasons, NV, SL, cadence)

    return NV, SL, BUD


def budseasons(bomin=0.01, bomax=0.08, bnmin=0.01, bnmax=0.15):

    budol = np.linspace(bomin, bomax, 100)
    budnl = np.linspace(bnmin, bnmax, 100)

    budo, budn = np.meshgrid(budol, budnl)

    ns = budn*(1.-budo)/(budo*(1.-budn))
    #ns = budn*(1.-budo)/(1.-budn*budo)
    return budo, budn, ns


def plotcontour(ax, Nfields, Nseasons, cadence, nvmin, nvmax, slmin, slmax, color='k', ls='-'):

    NV, SL, BUD = budplot(Nfields, Nseasons, cadence,
                          nvmin, nvmax, slmin, slmax)
    #plt.figure(figsize=(8, 6))
    # ax.imshow(BUD, extent=(
    #    nvmin, nvmax, slmin, slmax), aspect='auto', alpha=0.25, cmap='hsv')

    zzv = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    #zzv = [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    zzv = [0.03, 0.05, 0.08, 0.10, 0.15]
    zzvc = [100.*zz for zz in zzv]
    CS = plt.contour(NV, SL, BUD, zzv, colors=color,
                     linestyles=ls, linewidths=3)

    fmt = {}
    strs = ['$%3.2f$' % zz for zz in zzv]
    #strs = ['{}%'.format(np.int(zz)) for zz in zzvc]
    for l, s in zip(CS.levels, strs):
        fmt[l] = s
    ax.clabel(CS, inline=True, fontsize=15,
              colors=color, fmt=fmt)


Nfields = 5
Nseasons = 2
cadence = 1
nvmin, nvmax = 5, 250
slmin, slmax = 120, 240

coords = [(12.41, 180), (41.5, 180.), (61.05, 180.), (208.05, 180.)]
fig, ax = plt.subplots(figsize=(10, 10))
plotcontour(ax, Nfields, Nseasons, cadence, nvmin, nvmax, slmin, slmax)
#coords = [62.0, 208.05]
coords = [36.46, 208.05]
ax.plot(coords, [180.]*2,  'ko', markersize=9.)


Nseasons = 10
plotcontour(ax, Nfields, Nseasons, cadence, nvmin,
            nvmax, slmin, slmax, color='r', ls='dashed')
#coords = [12.41, 41.5]
coords = [7.3, 41.5]
ax.plot(coords, [180.]*2,  'ro', markersize=9.)


ax.set_xlabel('$\mathrm{N_{visits}}$', weight='normal')
ax.set_ylabel('Season length [days]', weight='normal')
# ax.grid(alpha=0.3)
ax.grid()
"""
Nfields = 5
Nseasons=10
cadence = 1
NVb,SLb, BUDb = budplot(Nfields,Nseasons,cadence,nvmin,nvmax,slmin,slmax)
CSb = plt.contour(NVb,SLb,BUDb,colors='b')
plt.clabel(CSb, inline=True, fontsize=10,colors='b')
"""

"""
fig, ax = plt.subplots()
ax.contour(NV,SL,Bud)
"""

fig, ax = plt.subplots()

bomin = 0.01
bomax = 0.08
bnmin = 0.03
bnmax = 0.15
# bnmin=2.
# bnmax=15.

budo, budn, ns = budseasons(bomin, bomax, bnmin, bnmax)
plt.imshow(ns, extent=(
    bomin, bomax, bnmin, bnmax), aspect='auto', alpha=0.25, cmap='hsv')
zzv = range(2, 11)
CSb = ax.contour(budo, budn, ns, zzv, colors='k')
fmt = {}
strs = ['$%i$' % zz for zz in zzv]
for l, s in zip(CSb.levels, strs):
    fmt[l] = s
plt.clabel(CSb, inline=True, fmt=fmt, fontsize=10, colors='k')
plt.xlabel('Budget (1 season/field)')
plt.ylabel('Budget')
plt.show()
