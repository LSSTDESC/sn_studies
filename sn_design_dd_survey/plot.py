import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .wrapper import Mod_z
import tkinter as tk
#from tkinter import tkFont
from tkinter import font as tkFont
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
matplotlib.use('tkagg')


class ShowVisits:
    def __init__(self, name, cadence=1., zmin=0.1, zmax=0.85):

        self.cadence = cadence
        self.zmin = zmin
        self.zmax = zmax
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

        self.z_nvisits = self.interp_z(sel['Nvisits'], sel['z'])

        self.gui()

    def interp_z(self, x, y):
        return interp1d(x, y, bounds_error=False, fill_value=0.)

    def plotNvisits(self):

        zstep = 0.005
        zvals = np.arange(self.zmin, self.zmax+zstep, zstep)

        self.ax.plot(zvals, self.dictvisits['nvisits']
                     (zvals), color='k', label='sum')

        for io, b in enumerate(self.bands):
            key = 'nvisits_{}'.format(b)
            self.ax.plot(zvals, self.dictvisits[key](
                zvals), color=self.colors[b], label='${}$-band'.format(b))
        self.ax.grid()
        self.ax.set_xlabel('z')
        self.ax.set_ylabel('Nvisits')
        self.ax.legend()
        self.ax.set_ylim(0,)

    def plotzlim(self, z=0.6):

        fontsize = 15

        ylims = self.ax.get_ylim()
        nvisits = int(np.round(self.dictvisits['nvisits'](z)))
        yref = 0.9*ylims[1]
        scale = 0.1*ylims[1]
        self.ax.text(0.3, yref, 'Nvisits={}'.format(
            nvisits), fontsize=fontsize)
        for io, b in enumerate(self.bands):
            key = 'nvisits_{}'.format(b)
            nvisits_b = int(np.round(self.dictvisits[key](z)))
            self.ax.text(0.3, 0.8*ylims[1]-scale*io,
                         'Nvisits - ${}$ ={}'.format(b, nvisits_b), fontsize=fontsize, color=self.colors[b])

        self.ax.text(0.95*z, 1.5*nvisits,
                     'z = {}'.format(z), fontsize=fontsize)
        self.ax.arrow(z, 1.4*nvisits, 0., -1.4*nvisits,
                      length_includes_head=True, color='r',
                      head_length=5, head_width=0.01)
        self.ax.set_ylim(0,)

    def plotnvisits(self, nvisits):

        # get the redshift from the total number of visits

        zlim = self.z_nvisits(nvisits)
        self.plotzlim(np.round(zlim, 2))

        self.ax.plot(self.ax.get_xlim(), [nvisits]*2,
                     color='r', linestyle='--')

    def gui(self, z=0.6):

        root = tk.Tk()
        self.fig = plt.Figure(figsize=(15, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        leg = 'days$^{-1}$'
        self.fig.suptitle('cadence: {} {}'.format(int(self.cadence), leg))
        self.fig.subplots_adjust(right=0.8)
        self.ax.set_xlim(self.zmin, self.zmax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        # self.ax.cla()
        self.plotNvisits()

        # common font
        helv36 = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # building the GUI
        # frame
        button_frame = tk.Frame(master=root, bg="white")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, expand=False)
        button_frame.place(relx=.9, rely=.5, anchor="c")
        # entries
        ents = self.make_entries(button_frame, font=helv36)

        # buttons
        heightb = 3
        widthb = 6

        nvisits_button = tk.Button(
            button_frame, text="Nvisits", command=(lambda e=ents: self.updateData_z(e)),
            bg='yellow', height=heightb, width=widthb, fg='blue', font=helv36)

        z_button = tk.Button(button_frame, text="zlim", command=(
            lambda e=ents: self.updateData_nv(e)), bg='yellow', height=heightb, width=widthb, fg='red', font=helv36)

        quit_button = tk.Button(button_frame, text="Quit",
                                command=root.quit, bg='yellow',
                                height=heightb, width=widthb, fg='black', font=helv36)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        nvisits_button.grid(row=2, column=0, sticky=tk.W+tk.E)
        z_button.grid(row=2, column=1, sticky=tk.W+tk.E)
        quit_button.grid(row=2, column=2, sticky=tk.W+tk.E)

        root.mainloop()

    def make_entries(self, frame, font):

        tk.Label(frame, text='zlim', bg='white',
                 fg='blue', font=font).grid(row=0)
        tk.Label(frame, text='Nvisits', bg='white',
                 fg='red', font=font).grid(row=1)

        entries = {}
        entries['zlim'] = tk.Entry(frame, width=5, font=font)
        entries['Nvisits'] = tk.Entry(frame, width=5, font=font)
        # entries['zlim'].pack(ipady=3)
        entries['zlim'].insert(10, "0.7")
        entries['Nvisits'] .insert(10, "50")
        entries['zlim'].grid(row=0, column=1)
        entries['Nvisits'] .grid(row=1, column=1)

        return entries

    def updateData_z(self, entries):

        self.ax.cla()
        self.plotNvisits()

        z = float(entries['zlim'].get())
        if z > 0:
            self.plotzlim(z=z)
        # update canvas
        self.ax.set_xlim(self.zmin, self.zmax)
        self.canvas.draw()

    def updateData_nv(self, entries):

        self.ax.cla()
        self.plotNvisits()

        nv = float(entries['Nvisits'].get())
        if nv > 0:
            self.plotnvisits(nvisits=nv)
        # update canvas
        self.ax.set_xlim(self.zmin, self.zmax)
        self.canvas.draw()
