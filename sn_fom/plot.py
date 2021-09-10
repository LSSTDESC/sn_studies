import matplotlib.pyplot as plt
import numpy as np


class plotStat:

    def __init__(self, params):

        self.params = params

    def plotFoM(self):

        io = -1
        r = []
        for index, row in self.params.iterrows():
            io += 1
            fom, rho = self.getFoM(row)
            r.append((io, fom, rho))

        res = np.rec.fromrecords(r, names=['iter', 'FoM', 'correl'])
        print(res)

        fig, ax = plt.subplots()
        ax.plot(res['iter'], res['FoM'])
        plt.show()

    def getFoM(self, params_fit):

        # get FoM
        sigma_w0 = np.sqrt(params_fit['Cov_w0_w0'])
        sigma_wa = np.sqrt(params_fit['Cov_wa_wa'])
        sigma_w0_wa = params_fit['Cov_w0_wa']

        fom, rho = FoM(sigma_w0, sigma_wa, sigma_w0_wa)

        return fom, rho


def FoM(sigma_w0, sigma_wa, sigma_w0_wa, coeff_CL=6.17):
    """
    Function to estimate the Figure of Merit (FoM)
    It is inversely proportional to the area of the error ellipse in the w0-wa plane

    Parameters
    ---------------
    sigma_w0: float
      w0 error
    sigma_wa: float
      wa error
    sigma_w0_wa: float
      covariance (w0,wa)
    coeff_CL: float, opt
      confidence level parameter for the ellipse area (default: 6.17=>95% C.L.)

    Returns
    ----------
    FoM: the figure of Merit
    rho: correlation parameter (w0,wa)


    """

    rho = sigma_w0_wa/(sigma_w0*sigma_wa)
    # get ellipse parameters
    a, b = ellipse_axis(sigma_w0, sigma_wa, sigma_w0_wa)
    area = coeff_CL*a*b

    return 1./area, rho


def ellipse_axis(sigx, sigy, sigxy):
    """
    Function to estimate ellipse axis

    Parameters
    ---------------
    sigx: float
      sigma_x
    sig_y: float
      sigma_y
    sigxy: float
      sigma_xy correlation

    Returns
    ----------
    (a,b) The two ellipse axis

    """

    comm_a = 0.5*(sigx**2+sigy**2)
    comm_b = 0.25*(sigx**2-sigy**2)**2-sigxy**2
    a_sq = comm_a+np.sqrt(comm_b)
    b_sq = comm_a-np.sqrt(comm_b)

    print('ellipse', sigx, sigy, sigxy, comm_a, comm_b)

    return np.sqrt(a_sq), np.sqrt(b_sq)
