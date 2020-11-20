import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sn_tools.sn_obs import season as seasoncalc
from sn_tools.sn_rate import SN_Rate
import healpy as hp
from optparse import OptionParser


class OS_Summary:
    """
    class to estimate the parameters of an Observing Strategy

    Parameters
    ---------------
    dbDir: str
       data directory
    dbName: str
       OS name
    prefix: str
      prefix for the filename 
    fieldName: str
       name of the field

    """

    def __init__(self, dbDir, dbName, fieldName, prefix):

        self.dbDir = dbDir
        self.dbName = dbName
        self.fieldName = fieldName
        self.fileName = '{}_{}_{}.npy'.format(dbName, prefix, fieldName)

        # this is to estimate the number of expected supernovae
        # get nside here
        ll = prefix.split('_')
        nside = int(ll[ll.index('nside')+1])
        self.pixArea = hp.nside2pixarea(nside, degrees=True)

        # SN rate here
        self.rateSN = SN_Rate(min_rf_phase=-15., max_rf_phase=45.)
        self.zmin = 0.01
        self.zmax = 0.5
        self.dz = 0.01

        # get the data
        tab = pd.DataFrame(self.load())
        print(tab.columns)
        npixels = len(np.unique(tab['healpixID']))
        print(npixels, len(tab))

        # getting seasons
        df = pd.DataFrame(np.copy(seasoncalc(tab.to_records(index=False))))

        # coadd observations (per night and per filter)
        dfcoadd = df.groupby(['healpixID']).apply(
            lambda x: self.calc(x)).reset_index()

        # estimate some stat on coadds
        finalres = dfcoadd.groupby(['healpixID', 'season']).apply(
            lambda x: self.stat(x)).reset_index()
        finalres = finalres.fillna(value=0)  # set NaNs to 0

        # add the estimated number of supernovae
        finalres['nSN'] = finalres['season_length'].apply(
            lambda x: self.nSN(x))
        idf = finalres['healpixID'] == 27238
        print(finalres[idf][['healpixID', 'season', 'season_length', 'nSN']])
        print(finalres['nSN'].sum())

    def load(self):
        """
        Method to load data from file

        Returns
        -----------
        numpy array of data

        """
        fi = '{}/{}/{}'.format(self.dbDir, self.dbName, self.fileName)
        print(fi)
        res = np.load(fi, allow_pickle=True)
        return res

    def calc(self, grp):
        """
        Method to estimate coadded values (per night) for a df group

        Parameters
        ---------------
        grp: pandas df group
          data to process

        Returns
        -----------
        pandas df with coadded values

        """
        # coaddition per night
        gb = grp.groupby(['night', 'filter']).apply(
            lambda x: self.coadd(x)).reset_index()

        return gb

    def coadd(self, grp):
        """
        Method to estimate coadded values (per night) for a df group

        Parameters
        ---------------
        grp: pandas df group
          data to process

        Returns
        -----------
        pandas df with coadded values
        """

        dictres = {}
        # mean values for these variables
        var_mean = ['pixRA', 'pixDec', 'observationStartMJD',
                    'fieldRA', 'fieldDec', 'seeingFwhmEff', 'fiveSigmaDepth', 'season']

        # sum for these
        var_sum = ['visitExposureTime', 'numExposures']

        for var in var_mean:
            dictres[var] = grp[var].mean()

        for var in var_sum:
            dictres[var] = grp[var].sum()

        # correct for 5-sigma depth
        dictres['fiveSigmaDepth'] += 1.25*np.log10(dictres['numExposures'])

        # this has to be done to be converted in pandas df
        dictb = {}
        for key, vals in dictres.items():
            dictb[key] = []
            dictb[key].append(vals)

        # return result
        return pd.DataFrame(dictb)

    def stat(self, grp):
        """
        Method to estimate some stat

        Parameters
        ---------------
        grp : pandas df group
          data to process

        Returns
        -----------
        pandas df with estimated stat parameters
        """
        dictres = {}

        for ff in np.unique(grp['filter']):
            io = grp['filter'] == ff
            sel = grp[io]
            dictres['N_{}'.format(ff)] = [int(np.median(sel['numExposures']))]
            dictres['m5_{}'.format(ff)] = [np.median(sel['fiveSigmaDepth'])]

            dictres['nights'] = [len(np.unique(grp['night']))]

        grp = grp.sort_values(by=['observationStartMJD'])

        season_min = np.min(grp['observationStartMJD'])
        season_max = np.max(grp['observationStartMJD'])
        dictres['season_length'] = [season_max-season_min]
        nights = np.sort(grp['night'].unique())
        cadence = 0.
        maxgap = 0.
        medgap = 0.
        if len(nights) >= 2:
            diff = np.asarray(nights[1:]-nights[:-1])
            cadence = np.median(diff).item()
            maxgap = np.max(diff).item()
            medgap = np.median(diff[diff > cadence]).item()

        dictres['cadence'] = [cadence]
        dictres['maxgap'] = [maxgap]
        dictres['medgap'] = [medgap]

        return pd.DataFrame(dictres)

    def nSN(self, duration):
        """
        Method to estimate the number of supernovae

        Parameters
        ---------------
        duration: float
          duration of the survey

        Returns
        -----------
        the total number of supernovae
        """
        #duration = grp['season_length']
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(
            zmin=self.zmin, zmax=self.zmax, dz=self.dz, bins=None,
            account_for_edges=True,
            duration=duration, survey_area=self.pixArea)

        nsn_sum = np.cumsum(nsn)

        # return pf.DataFrame({'nSN': nsn_sum})
        return nsn_sum[-1]


parser = OptionParser()

parser.add_option("--dbName", type="str",
                  default='descddf_v1.5_10yrs', help="db name [%default]")
parser.add_option("--dbDir", type="str",
                  default='/media/philippe/LSSTStorage/ObsPixelized', help="db dir [% default]")
parser.add_option("--fieldNames", type="str", default='COSMOS',
                  help=" list of fieldNames  [%default]")
parser.add_option("--prefix", type="str",
                  default='DD_nside_64_0.0_360.0_-1.0_-1.0', help="file name prefix [%default]")

opts, args = parser.parse_args()

fieldNames = opts.fieldNames.split(',')

for fieldName in fieldNames:
    OS_Summary(opts.dbDir, opts.dbName, fieldName, opts.prefix)
