from sn_tools.sn_telescope import Telescope
import numpy as np
from scipy import interpolate
from . import plt
from scipy.interpolate import interp1d
from .wrapper import Mod_z
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def flux5_to_m5(bands):
    """
    Function to estimate m5 from 5-sigma fluxes

    Parameters
    ----------
    bands: str
     filters considered

    Returns
    -------
    f5_dict: dict
     keys = bands
     values = interp1d(flux_5,m5)
    """

    m5_range = np.arange(20., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    f5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range))[:, 1]
        f5_dict[band] = interpolate.interp1d(
            flux_5, m5_range, bounds_error=False, fill_value=0.0)

    return f5_dict


def m5_to_flux5(bands):
    """
    Function to estimate 5-sigma fluxes from m5 values

    Parameters
    ----------
    bands: str
     filters considered

    Returns
    -------
    f5_dict: dict
     keys = bands
     values = interp1d(m5,flux_5)
    """
    m5_range = np.arange(20., 28.0, 0.01)

    # estimate the fluxes corresponding to this range

    telescope = Telescope(airmass=1.2)

    m5_dict = {}
    for band in bands:
        flux_5 = telescope.mag_to_flux_e_sec(
            m5_range, [band]*len(m5_range), [30.]*len(m5_range))[:, 1]
        m5_dict[band] = interpolate.interp1d(
            m5_range, flux_5, bounds_error=False, fill_value=0.0)

    return m5_dict


class ShowVisits:
    def __init__(self, name, cadence=1.):

        self.cadence = cadence
        data = Mod_z(
            'Nvisits_cadence_Nvisits_median_m5_filter.npy').nvisits

        # select data for this cadence
        idx = np.abs(data['cadence']-self.cadence) < 1.e-5
        sel = data[idx]

        # prepare interpolators for plot
        self.zmin = np.min(sel['z'])
        self.zmax = np.max(sel['z'])
        self.dictvisits = {}
        self.dictvisits['nvisits'] = self.interp_z(sel['z'], sel['Nvisits'])

        self.bands = 'grizy'
        self.colors = dict(zip(self.bands, ['c', 'g', 'y', 'r', 'm']))
        for b in self.bands:
            self.dictvisits['nvisits_{}'.format(b)] = self.interp_z(
                sel['z'], sel['Nvisits_{}'.format(b)])

    def interp_z(self, x, y):
        return interp1d(x, y, bounds_error=False, fill_value=0.)

    def plotNvisits(self, ax):

        zstep = 0.005
        zvals = np.arange(self.zmin, self.zmax+zstep, zstep)

        ax.plot(zvals, self.dictvisits['nvisits']
                (zvals), color='k', label='sum')

        for io, b in enumerate(self.bands):
            key = 'nvisits_{}'.format(b)
            ax.plot(zvals, self.dictvisits[key](
                zvals), color=self.colors[b], label='${}$-band'.format(b))
        ax.grid()
        ax.set_xlabel('z')
        ax.set_ylabel('Nvisits')
        ax.legend()
        ax.set_ylim(0,)

    def plotzlim(self, ax, z=0.6):

        fontsize = 15

        ylims = ax.get_ylim()
        nvisits = int(np.round(self.dictvisits['nvisits'](z)))
        yref = 0.9*ylims[1]
        scale = 0.1*ylims[1]
        ax.text(0.3, yref, 'Nvisits={}'.format(nvisits), fontsize=fontsize)

        for io, b in enumerate(self.bands):
            key = 'nvisits_{}'.format(b)
            nvisits_b = int(np.round(self.dictvisits[key](z)))
            ax.text(0.3, 0.8*ylims[1]-scale*io,
                    'Nvisits - ${}$ ={}'.format(b, nvisits_b), fontsize=fontsize, color=self.colors[b])

        ax.text(0.95*z, 1.5*nvisits, 'z = {}'.format(z), fontsize=fontsize)
        ax.arrow(z, 1.4*nvisits, 0., -1.4*nvisits,
                 length_includes_head=True, color='r',
                 head_length=5, head_width=0.01)
        ax.set_ylim(0,)

    def __call__(self, z=0.6):

        root = tk.Tk()

        # fig, ax = plt.subplots(figsize=(11, 6))

        fig = plt.Figure(figsize=(11, 6), dpi=100)
        ax = fig.add_subplot(111)
        leg = 'day$^{-1}$'
        fig.suptitle('cadence: {} {}'.format(int(self.cadence), leg))
        self.plotNvisits(ax)

        bar1 = FigureCanvasTkAgg(fig, root)
        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        self.updateData(ax, -1)
        B = tk.Button(root, text="Get visits",
                      command=self.updateData(ax, z=-1))

        # B.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        B.pack()
        entree = tk.Entry(root, textvariable='string', width=30)
        entree.pack()
        """
        tk.Canvas(root, width=250, height=50, bg='ivory').pack(
            side=tk.LEFT, padx=5, pady=5)
        
        for line, item in enumerate(['foo', 'bar', 'stuff']):
            l = tk.Label(root, text=item, width=10)
            e = tk.Entry(root, width=10)
            l.grid(row=line, column=0)
            e.grid(row=line, column=1)
        """
        root.mainloop()

    def updateData(self, ax, z):

        ax.clear()
        self.plotNvisits(ax)

        if z > 0:
            self.plotzlim(ax, z=z)

    def call(self, z=0.5):

        colors = dict(zip(self.bands, ['c', 'g', 'y', 'r', 'm']))
        fontsize = 15
        zstep = 0.005
        zvals = np.arange(self.zmin, self.zmax+zstep, zstep)

        fig, ax = plt.subplots(figsize=(11, 6))
        leg = 'day$^{-1}$'
        fig.suptitle('cadence: {} {}'.format(int(self.cadence), leg))
        ax.plot(zvals, self.dictvisits['nvisits']
                (zvals), color='k', label='sum')

        ylims = ax.get_ylim()
        nvisits = int(np.round(self.dictvisits['nvisits'](z)))
        yref = 0.9*ylims[1]
        scale = 0.1*ylims[1]
        ax.text(0.3, yref, 'Nvisits={}'.format(nvisits), fontsize=fontsize)
        for io, b in enumerate(self.bands):
            key = 'nvisits_{}'.format(b)
            nvisits_b = int(np.round(self.dictvisits[key](z)))
            ax.text(0.3, 0.8*ylims[1]-scale*io,
                    'Nvisits - ${}$ ={}'.format(b, nvisits_b), fontsize=fontsize, color=colors[b])
            ax.plot(zvals, self.dictvisits[key](
                zvals), color=colors[b], label='${}$-band'.format(b))

        ax.text(0.95*z, 1.5*nvisits, 'z = {}'.format(z), fontsize=fontsize)
        ax.arrow(z, 1.4*nvisits, 0., -1.4*nvisits,
                 length_includes_head=True, color='r',
                 head_length=5, head_width=0.01)
        ax.grid()
        ax.set_xlabel('z')
        ax.set_ylabel('Nvisits')
        ax.legend()
        ax.set_ylim(0,)
