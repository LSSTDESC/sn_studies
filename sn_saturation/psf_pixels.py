import numpy as np
from . import plt
import numpy.lib.recfunctions as rf
import time
import multiprocessing
from scipy.integrate import dblquad, quad
from astropy.table import Table
import pandas as pd


class PSF_pixels:
    """
    class to estimate the max flux fraction in a pixel
    depending on a psf

    Parameters
    ---------------
    seeing: float
      seeing value
    psf_type: str
      type of the psf
    scanfast: bool, opt
      bool to decide the grid of the flux position center scan (default: True)


    """

    def __init__(self, seeing, psf_type, scanfast=True):

        self.seeing = seeing
        self.seeing_pixel = seeing/0.2  # 1 LSST pixel~ 0.2"
        self.flux = 1000.
        self.scanfast = scanfast

        if psf_type == 'single_gauss':
            self.PSF = self.PSF_single_gauss
        if psf_type == 'double_gauss':
            self.PSF = self.PSF_double_gauss
        if psf_type == 'moffat':
            self.PSF = self.PSF_moffat

    def PSF_single_gauss(self, x, y, xc, yc, nsigma=1):
        """
        Method to estimate a single gaussian PSF

        Parameters
        --------------
        x: float
          x-coordinate where the PSF has to be estimated
        y: float
          y-coordinate where the PSF has to be estimated
        xc: float
           x-center of the pixel
        yc: float
           y-center of the pixel
        sigma: float
          sigma of the gaussian

        Returns
        ---------
        flux in the pixel (float)

        """
        # sigma = self.seeing_pixel/2.355
        if isinstance(x, np.ndarray):
            val = (x[..., None]-xc)**2+(y[..., None]-yc)**2
            # val = (x[...,None]-xc)**2
        else:
            val = (x-xc)**2+(y-yc)**2
        sigma = nsigma*self.seeing_pixel/2.355
        func = np.exp(-val/2./sigma**2)
        func /= (2.*np.pi*sigma**2)
        # func /= (2.*np.pi)
        # func /= (sigma*np.sqrt(2.*np.pi))
        return self.flux*func

    def PSF_double_gauss(self, x, y, xc, yc):
        """
        Method to estimate a double-gaussian PSF

        Parameters
        --------------
        x: float
          x-coordinate where the PSF has to be estimated
        y: float
          y-coordinate where the PSF has to be estimated
        xc: float
           x-center of the pixel
        yc: float
           y-center of the pixel
        sigma: float
          sigma of the gaussian

        Returns
        ---------
        flux in the pixel (float)

        """
        return 0.909*(self.PSF_single_gauss(x, y, xc, yc)+0.1*self.PSF_single_gauss(x, y, xc, yc, 2))

    def PSF_moffat(self, x, y, xc, yc, beta=2.5):
        """
        Method to estimate a moffat PSF

        Parameters
        --------------
        x: float
          x-coordinate where the PSF has to be estimated
        y: float
          y-coordinate where the PSF has to be estimated
        xc: float
           x-center of the pixel
        yc: float
           y-center of the pixel
        beta: float,opt
          moffat beta value (default: 2.5)

        Returns
        ---------
        flux in the pixel (float)

        """
        # sigma = self.seeing_pixel/2.355
        if isinstance(x, np.ndarray):
            val = (x[..., None]-xc)**2+(y[..., None]-yc)**2
            # val = (x[...,None]-xc)**2
        else:
            val = (x-xc)**2+(y-yc)**2

        alpha = self.seeing_pixel
        vv = 2.*np.sqrt(2**(1/beta)-1)
        alpha /= vv
        func = val/alpha**2
        func = func**beta
        func = 1./func
        # func /= (2.*np.pi)
        # func /= (sigma*np.sqrt(2.*np.pi))
        return self.flux*func

    def GetCenters(self, xmin, xmax, dx, ymin, ymax, dy):
        """
        Estimate centers of a grid defined by (xmin,xmax,ymin,ymax)

        Parameters:
        -----------
        xmin, xmax, dx: min x, max x, grid space in x
        ymin, ymax, dy: min y, max y, grid space in y

        Returns:
        --------
        xc, yc : coordinates of the center

        """
        X, Y = self.Mesh(xmin, xmax, dx, ymin, ymax, dy)
        xc = (X[:-1, 1:]+X[:-1, :-1])*0.5
        yc = (Y[1:, :-1]+Y[:-1, :-1])*0.5

        return xc, yc

    def Mesh(self, xmin, xmax, dx, ymin, ymax, dy):
        """
        Estimate a mesh grid

        P Parameters:
        -----------
        xmin, xmax, dx: min x, max x, grid space in x
        ymin, ymax, dy: min y, max y, grid space in y

        Returns:
        --------
        X,Y : mesh grid

        """

        nx = int((xmax-xmin)/dx)+1
        ny = int((ymax-ymin)/dy)+1
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def get_psf_flux(self, integ_type, xmin, xmax, dx, ymin, ymax, dy, xc, yc):
        """
        Estimate fluxes from PSF on a grid

        P Parameters:
        -----------
        xmin, xmax, dx: min x, max x, grid space in x
        ymin, ymax, dy: min y, max y, grid space in y
        xc, yc: flux coordinates

        Returns:
        --------
        X,Y : mesh grid

        """

        xpixel = 0.5*(xmin+xmax)
        ypixel = 0.5*(ymin+ymax)
        n_echant = 20
        dx_new = dx/n_echant
        dy_new = dy/n_echant

        xc_bary, yc_bary = self.GetCenters(
            xmin, xmax, dx_new, ymin, ymax, dy_new)
        # sigma = self.seeing_pixel/2.355
        if integ_type == 'num':
            flux_pixel = self.PSF(xc_bary, yc_bary, xc,
                                  yc)*dx_new*dy_new
        if integ_type == 'quad':
            flux_from_psf_vect = np.vectorize(self.integr)
            flux_pixel = flux_from_psf_vect(
                xmin, xmax, ymin, ymax, xc, yc)
            # flux_pixel = flux_from_psf_vect(xmin,xmax,0,0,xc,yc,sigma)

        if flux_pixel.ndim > 1:
            res = np.array(np.sum(flux_pixel, axis=(0, 1)) /
                           self.flux, dtype=[('pixel_frac', 'f8')])
        else:
            res = np.array(flux_pixel/self.flux, dtype=[('pixel_frac', 'f8')])

        res = rf.append_fields(res, 'xc', xc)
        res = rf.append_fields(res, 'yc', yc)
        res = rf.append_fields(res, 'seeing', [self.seeing]*len(res))
        res = rf.append_fields(res, 'xpixel', [xpixel]*len(res))
        res = rf.append_fields(res, 'ypixel', [ypixel]*len(res))
        res = rf.append_fields(res, 'dx', [dx]*len(res))
        res = rf.append_fields(res, 'dy', [dy]*len(res))

        return res

    def integr(self, xmin, xmax, ymin, ymax, xc, yc, sigma):
        """
        Flux integration over (xmin, xmax, ymin, ymax)


        Returns:
        --------
        numpy array of results

        """

        return dblquad(self.PSF, xmin, xmax, lambda x: ymin, lambda y: ymax,
                       args=(xc, yc, sigma))[0]
        """
        res = quad(lambda x : np.exp(-(x-xc)**2/2./sigma**2)/ \
                   np.sqrt(2.*np.pi)/sigma,xmin,xmax)[0]
        # print('hello',xmin,xmax,xc,yc,sigma,res)
        return self.flux*res
        """

    def get_flux_map(self, integ_type='num'):
        """
        Estimate fluxes from PSF

        Parameters
        ---------------
        integ_type: str,opt
          type of integration (num/quad - default: num)


        Returns
        ----------
        numpy array with the following fields:

        pixel_frac: fraction of signal in the pixel
        xc: x position of the flux
        yc: y position of the flux
        seein: seeing value
        xpixel: x coodrinate of the pixel center
        ypixel: y coordinate of the pixel center
        dx: pixel size in x
        dy: pixel size in y
        """
        # limits of the grid
        dx, dy = 1., 1.
        dgrid = int(2.15*self.seeing_pixel)  # 5 sigma
        # dgrid = int(5*self.seeing_pixel)  # 5 sigma
        # print('dgrid', dgrid, self.seeing)
        if dgrid < 3:
            dgrid = 5
        # dgrid = 1
        valgrid = (dgrid-1)
        xmin, xmax = -dx/2.-valgrid, dx/2.+valgrid
        ymin, ymax = -dy/2.-valgrid, dy/2.+valgrid

        # grid of flux (center) positions on the pixel
        dxc = 0.01
        dyc = 0.01
        xmin_c, xmax_c = -dx/2., dx/2
        ymin_c, ymax_c = -dy/2, dy/2
        nxc = int((xmax_c-xmin_c)/dxc)+1
        nyc = int((ymax_c-ymin_c)/dyc)+1
        xc = np.linspace(xmin_c, xmax_c, nxc)
        yc = np.linspace(ymin_c, ymax_c, nyc)

        if self.scanfast:
            xcm, ycm = np.mgrid[xmin_c:xmax_c:3j, ymin_c:ymax_c:3j]
        else:
            xcm, ycm = np.mgrid[xmin_c:xmax_c:10j, ymin_c:ymax_c:10j]
        positions = np.vstack([xcm.ravel(), ycm.ravel()])

        # pixel centers on the grid
        xc_grid, yc_grid = self.GetCenters(xmin, xmax, dx, ymin, ymax, dy)

        """
        plt.plot(xc_grid,yc_grid,'ko')
        plt.plot(positions[0,:],positions[1,:],'r*')
        plt.show()
        """

        xc_grid = np.unique(xc_grid)
        yc_grid = np.unique(yc_grid)

        positions_main = [(x, y) for x in xc_grid for y in yc_grid]
        restot = None

        time_ref = time.time()

        if len(xc_grid) > 1:
            nmulti = 8
            ntoprocess = len(positions_main)
            nperbatch = int(ntoprocess/nmulti)
            batches = range(0, ntoprocess, nperbatch)
            njobs = len(batches)-1
            if ntoprocess not in batches:
                batches = np.append(batches, ntoprocess)

            print('npos', ntoprocess)
            result_queue = multiprocessing.Queue()
            for j in range(njobs+1):
                imin = batches[j]
                imax = batches[j+1]
                pos = positions_main[imin:imax]
                p = multiprocessing.Process(name='Subprocess-'+str(j),
                                            target=self.loop_process, args=(integ_type, pos, dx, dy, positions[0], positions[1], j, result_queue))
                p.start()

            resultdict = {}
            for j in range(njobs+1):
                resultdict.update(result_queue.get())

            for p in multiprocessing.active_children():
                p.join()

            for j in range(njobs+1):
                if restot is None:
                    restot = resultdict[j]
                else:
                    restot = np.concatenate((restot, resultdict[j]))

        else:
            restot = self.loop_process(
                integ_type, positions_main, dx, dy, positions[0], positions[1])
        print('Done', time.time()-time_ref)
        return restot

    def loop_process(self, integ_type, positions, dx, dy, xc, yc, j=-1, output_q=None):
        """
        Loop over pixel centers to get the flux, depending on the position

        Parameters:
        -----------
        integ_type: type of integration ('num' or 'quad')
        positions: positions (x,y) of the center of the considered pixel
        dx, dy: pixel size in x and y
        xc, yc: coordinates of the flux center

        Returns:
        -------
        numpy array with the following fields:
        pixel_frac: fraction of signal in the pixel
        xc: x position of the flux
        yc: y position of the flux
        seein: seeing value
        xpixel: x coordinate of the pixel center
        ypixel: y coordinate of the pixel center
        dx: pixel size in x
        dy: pixel size in y



        """

        """
        plt.plot(positions[0],positions[1],'r*')
        plt.plot(xc,yc,'ko')
        plt.show()
        """

        res = None
        for (x, y) in positions:
            fluxes = self.get_psf_flux(
                integ_type, x-dx/2., x+dx/2., dx, y-dy/2., y+dy/2., dy, xc, yc)
            if res is None:
                res = fluxes
            else:
                res = np.concatenate((res, fluxes))

        if output_q is not None:
            output_q.put({j: res})
        else:
            return res


def PlotMaxFrac(psf_type='single_gauss', title='Single gaussian profile'):
    """
    Function to display max frac pixel vs seeing

    Parameters
    --------------
    psf_type: str, opt
      PSF type (default: single_gauss)
    title: str, opt
      title for the plot (default: Single gaussian profile)

    """

    tab = np.load('PSF_pixel_{}.npy'.format(psf_type))

    # first grab the pixels with the frac max flux (per seeing)

    df = pd.DataFrame(tab)
    min_seeing = df['seeing'].min()
    max_seeing = df['seeing'].max()

    df = df.round({'xpixel': 2, 'ypixel': 2, 'xc': 2, 'yc': 2, 'seeing': 2})

    # take the pixel in (0,0)
    idx = np.abs(df['xpixel']) < 1.e-5
    idx &= np.abs(df['ypixel']) < 1.e-5
    df = df[idx]

    grp = df.groupby(['seeing']).apply(lambda x: pd.DataFrame({'pixel_frac_max': [x['pixel_frac'].max()],
                                                               'pixel_frac_min': [x['pixel_frac'].min()],
                                                              'pixel_frac_med': [x['pixel_frac'].median()]})).reset_index()
    print(grp)

    # fontsize = 20
    fig, ax = plt.subplots(figsize=(12, 8))
    # ax.set_title(title,fontsize=fontsize)
    ax.set_title(title)
    ax.plot(grp['seeing'], grp['pixel_frac_med'],
            ls='-', color='r', linewidth=2)
    ax.fill_between(grp['seeing'], grp['pixel_frac_min'],
                    grp['pixel_frac_max'], alpha=0.5)
    # ax[i].set_ylim([0.,0.35])
    ax.set_xlim([min_seeing, max_seeing])
    ax.grid()
    """
    ax.set_xlabel(r'seeing ["]',fontsize=fontsize)
    ax.set_ylabel(r'Max frac pixel flux',fontsize=fontsize)
    ax.tick_params(labelsize = fontsize)
    """
    ax.set_xlabel(r'seeing ["]')
    ax.set_ylabel(r'Max frac pixel flux')
    # ax.tick_params(labelsize = fontsize)
    plt.savefig('max_frac_seeing_{}.png'.format(psf_type))


def PlotPixel(seeing, psf_type, varxsel, xp, varysel, yp, varxplot, varyplot, titleadd, type_plot='imshow'):
    """
    Display of the pixels fraction distribution

    Parameters:
    -----------
    seeing: seeing value
    res: numpy array of results
    varxsel: x variable to display
    xp: x of the flux center or of the pixel center
    varysel: y variable to display
    yp: y of the flux center or of the pixel center
    type_plot: two options:
    - imshow: display the fraction of signal distribution (all pixels)
    - contour: display the fraction of signal for one pixel (all flux position)
    Returns:
    -------
    None

    """

    res = np.load('PSF_pixel_{}.npy'.format(psf_type))
    print(res.dtype)
    fontsize = 20
    idx = np.abs(res['seeing']-seeing) < 1.e-3
    print(len(res[idx]))
    idx &= np.abs(res[varxplot]-xp) < 1.e-5
    idx &= np.abs(res[varyplot]-yp) < 1.e-5
    sel = res[idx]
    print('here man', varxplot, varyplot, seeing, len(sel))
    # print('selection',len(sel),np.unique(res[[varxsel,varysel]]))
    fig, ax = plt.subplots(figsize=(10, 10))
    seeing_pix = seeing/0.2  # seeing in arcsec - pixel LSST = 0.2"
    sigma = seeing_pix/2.355
    # titleform = 'seeing: {} - sigma: {} pixel'.format(np.round(seeing,2),np.round(sigma,2))
    titleform = '{} - seeing: {}"'.format(titleadd, np.round(seeing, 2))
    # fig.suptitle(titleform,fontsize=fontsize)
    # ax.set_title(titleform,fontsize=fontsize)
    ax.set_title(titleform)
    dim = int(np.sqrt(len(sel)))
    xcm = np.reshape(sel[varxsel], (dim, dim))
    ycm = np.reshape(sel[varysel], (dim, dim))
    Zc = np.reshape(sel['pixel_frac'], (dim, dim))

    print('hhh', np.min(xcm), np.max(xcm), dim, len(sel))
    if type_plot == 'contour':
        CS = ax.contourf(xcm, ycm, Zc, 20, cmap=plt.cm.viridis)
    if type_plot == 'imshow':
        CS = ax.imshow(Zc, extent=[np.min(xcm), np.max(
            xcm), np.min(ycm), np.max(ycm)])
    # ax.plot(xxc_m, yyc_m, 'k.')
    vmin = np.min(sel['pixel_frac'])
    vmax = np.max(sel['pixel_frac'])
    vmin = np.round(vmin, 2)
    vmax = np.round(vmax, 2)
    print(vmin, vmax, np.arange(vmin, vmax, 0.05))
    shrink = 0.82
    # shrink = 1.
    cbar = fig.colorbar(CS, ax=ax, ticks=np.arange(
        vmin, vmax, 0.01), shrink=shrink)
    """
    cbar.set_label('Flux fraction', rotation=270,
                   fontsize=fontsize,labelpad=30)# position=(12.,0.5))
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r'x [pixel]',fontsize=fontsize)
    ax.set_ylabel(r'y [pixel]',fontsize=fontsize)
    ax.tick_params(labelsize = fontsize)
    """
    cbar.set_label('Flux fraction', rotation=270,
                   labelpad=30)  # position=(12.,0.5))
    # cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(r'x [pixel]')
    ax.set_ylabel(r'y [pixel]')
    # ax.tick_params(labelsize = fontsize)
    ax = plt.gca()
    ax.set_aspect('equal')
    # ax.set_xlim([-3.,3.])
    # ax.set_ylim([-3.,3.])
    # cbar.set_clim(np.round(vmin,1), np.round(vmax,1))
    # plt.show()
    plt.savefig('flux_dist_center_position.png')


def test_newmethod():
    def PSF_single_gauss(x, y, xc, yc, sigma):
        """
        Method to estimate a single gaussian PSF

        Parameters
        --------------
        x: float
        x-coordinate where the PSF has to be estimated
        y: float
        y-coordinate where the PSF has to be estimated
        xc: float
        x-center of the pixel
        yc: float
        y-center of the pixel
        sigma: float
        sigma of the gaussian

        Returns
        ---------
        flux in the pixel (float)

        """
        # sigma = self.seeing_pixel/2.355
        val = (x-xc)**2+(y-yc)**2

        func = np.exp(-val/2./sigma**2)
        func /= (2.*np.pi*sigma**2)
        # func /= (2.*np.pi)
        # func /= (sigma*np.sqrt(2.*np.pi))
        return func

    dx = 0.1
    dy = 0.1
    pixel_LSST = 0.2

    seeing = 0.5
    seeing_pixel = seeing/pixel_LSST
    sigma = seeing_pixel/2.355

    nsize = int(5.*sigma)

    x = np.arange(-nsize+dx, nsize, dx)
    y = np.arange(-nsize+dy, nsize, dy)

    xtile = np.tile(x, (len(y), 1))
    ytile = np.tile(y, (len(x), 1)).transpose()

    xc = 0.
    yc = 0.
    fluxes = PSF_single_gauss(xtile, ytile, xc, yc, sigma)*dx*dy

    plt.plot(xtile, ytile, 'ko')
    plt.show()

    # for i in range(-nsize,nsize):
    #    for j in range(-nsize,nsize):
    for ix in range(len(x)-1):
        for iy in range(len(y)-1):
            idx = xtile >= x[ix]
            idx &= xtile < x[ix+1]
            idy = ytile >= y[iy]
            idy &= ytile < y[iy+1]
            print(0.5*(x[ix]+x[ix+1]), 0.5*(y[iy]+y[iy+1]),
                  np.sum(fluxes[idx & idy]))

    print(fluxes)
