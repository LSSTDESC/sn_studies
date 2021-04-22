import numpy as np
from optparse import OptionParser
from sn_tools.sn_obs import season


def refVals(tab, filterCol='band', m5Col='fiveSigmaDepth',
            RACol='RA', DecCol='Dec'):
    """
    Function to extract ref values (m5, RA, Dec) from tab

    Parameters
    ---------------
    tab: numpy array
      data to extract ref vals from
    filterCol: str, opt
       name of the filter column (default: band)
    m5Col: str, opt
      name of the 5-sigma depth col (default: fiveSigmaDepth)
    RACol: str, opt
      name of the RA col (default: RA)
    DecCol: str, opt
      name of the Dec col (default: Dec)
    """

    outdict = {}
    for season in np.unique(tab['season']):
        idx = tab['season'] == season
        sel = tab[idx]
        outdict[season] = {}
        for b in np.unique(sel[filterCol]):
            idxb = sel[filterCol] == b
            outdict[season]['m5_{}'.format(b)] = np.median(sel[idxb][m5Col])
        outdict[season][RACol] = np.mean(sel[RACol])
        outdict[season][DecCol] = np.mean(sel[DecCol])

    return outdict


def getRefs(dirFile, dbName, fieldNames):
    """
    Function to get refs for all the fields
    """
    fName = '{}/{}.npy'.format(dirFile, dbName)

    obs_ref = np.load(fName, allow_pickle=True)

    refs = {}
    for fieldName in fieldNames:
        refs[fieldName] = {}
        idx = obs_ref['note'] == 'DD:{}'.format(fieldName)
        sel = season(obs_ref[idx], mjdCol='mjd')
        refs[fieldName] = refVals(sel)

    return refs


class Observation:
    """
    class to handle observation

    Parameters
    ---------------
    varnames: list(str), opt
      list of variables defining an observation

    """

    def __init__(self, varnames=['observationId', 'RA', 'Dec', 'mjd', 'flush_by_mjd', 'exptime', 'band', 'rotSkyPos', 'numExposures', 'airmass', 'seeingFwhm500', 'seeingFwhmEff', 'seeingFwhmGeom', 'sky', 'night', 'slewTime', 'visitTime', 'slewDistance', 'fiveSigmaDepth', 'altitude', 'azimuth', 'paraAngle', 'cloud', 'moonAlt', 'sunAlt', 'note', 'fieldId', 'proposalId', 'block_id', 'observationStartLST', 'rotTelPos', 'moonAz', 'sunAz', 'sunRA', 'sunDec', 'moonRA', 'moonDec', 'moonDistance', 'solarElong', 'moonPhase', 'season']):

        self.varnames = varnames
        self.data = {}
        for v in varnames:
            self.data[v] = 0.0

    def fill(self, varname, val):
        """
        Method to fill the dict corresponding to an observation

        Parameters
        ---------------
        varname: str 
          variable name
        val: float
          value for this variable

        """
        if varname not in self.varnames:
            print('Problem: this variable is not part of observation:', varname)
            return
        self.data[varname] = val


class DitheredObs:
    """
    class to generate dithered observations

    Parameters
    ---------------
    refs: dict, opt
      dict of ref values (RA, Dec, m5_g,m5_r,m5_i,m5_z,m5_y)
    seasons: dict, opt
      dict of infos on seasons (season num, season length) to generate
    cadence: float, opt
      cadence for observations (default: 1.)
    dither: float, opt
      translational dither offset (default: 0.0)
    filteralloc: dict, opt
      visits per band and per obs night


    """

    def __init__(self, refs=dict(zip([1], [dict(zip(['RA', 'Dec', 'm5_g', 'm5_r', 'm5_i', 'm5_z', 'm5_y'],
                                                    [150.0, 2.18, 24.49, 24.04, 23.6, 22.98, 22.14]))])),
                 seasons=dict(zip([1], [180.])),
                 cadence=1.,
                 dither=0.0,
                 filteralloc=dict(zip('grizy', [10, 20, 20, 26, 20]))):

        self.refs = refs
        self.seasons = seasons
        self.cadence = cadence
        self.dither = dither
        self.filteralloc = filteralloc
        self.mjdmin = 59954.  # october 1rst, 2023
        self.interseason = 100  # 100 days between seasons
        self.numExposures = 1  # 1 snap
        self.exposureTime = 30.  # exposure time in sec
        self.visitTime = 31.  # visit time in sec
        self.airmass = 1.2  # airmass common value

    def generateObs(self):
        """
        Method to generate observations

        """
        obs = None
        for key, vals in self.seasons.items():
            valsref = self.refs[key]
            res = self.generateSeason(key, vals, valsref)
            if obs is None:
                obs = res
            else:
                obs = np.concatenate((obs, res))

        return obs

    def generateSeason(self, seasnum, season_length, valsref):
        """
        Method to generate observations for a season

        Parameters
        ---------------
        seasnum: int
          season number
        season_length: float
          season length [days]
        valsref: dict
          dict of reference values (m5, RA, Dec)

        """

        obs = Observation()
        # MJD min of the season
        mjd_min = self.mjdmin+(seasnum-1)*self.interseason

        # MJD of observing nights
        mjd_days = np.arange(
            mjd_min, mjd_min+season_length+self.cadence, self.cadence)

        obsID = 0
        r = []
        print('valsref', valsref)
        for mjd in mjd_days:
            night = int(mjd-mjd_min)+1
            # make the sequence for each filter
            angle = np.round(np.random.uniform(0., 2.*np.pi), 2)
            RA_d = valsref['RA']+self.dither*np.cos(angle)
            Dec_d = valsref['Dec']+self.dither*np.sin(angle)
            obs.fill('RA', RA_d)
            obs.fill('Dec', Dec_d)
            obs.fill('numExposures', self.numExposures)
            obs.fill('airmass', self.airmass)
            obs.fill('night', night)
            obs.fill('season', seasnum)
            obs.fill('visitTime', self.visitTime)
            obs.fill('exptime', self.exposureTime)

            for key, vals in self.filteralloc.items():
                for nn in range(vals):
                    obsID += 1
                    mmjd = mjd+nn*self.exposureTime/(24.*3600.)
                    obs.fill('band', key)
                    obs.fill('fiveSigmaDepth', valsref['m5_{}'.format(key)])
                    obs.fill('observationId', obsID)
                    obs.fill('mjd', mmjd)
                    # print(obs.data)
                    r.append(list(obs.data.values()))
        # print(r)
        return np.rec.fromrecords(r, names=obs.varnames)


def outName(dither, cadence):

    return 'Fakes_dither{}_cadence_{}.npy'.format(np.round(dither, 2), cadence)


parser = OptionParser(
    description='generate a set of dithered observations')
parser.add_option("--dirFile", type="str",
                  default='../../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str", default='descddf_v1.5_10yrs',
                  help="dbName to get ref values[%default]")
parser.add_option("--fieldNames", type="str", default='COSMOS',
                  help="fields to process [%default]")
parser.add_option("--cadence", type=str, default='1.',
                  help="cadence of observation [%default]")
parser.add_option("--dither", type=str, default='0.0',
                  help="translational dither offset [%default]")

opts, args = parser.parse_args()

fieldNames = opts.fieldNames.split(',')
cadence = list(map(float, opts.cadence.split(',')))
dither = list(map(float, opts.dither.split(',')))

# load the database to get ref values (m5, RA, Dec, ...)

refVals = getRefs(opts.dirFile, opts.dbName, fieldNames)

print('eee', cadence, dither)
for fieldName in fieldNames:
    for cad in cadence:
        for dith in dither:
            obs = DitheredObs(
                refs=refVals[fieldName], cadence=cad, dither=dith)

            allobs = obs.generateObs()

            # print(allobs.dtype)
            np.save(outName(dith, cad), allobs)
            nights = np.unique(allobs['night'])
            print('cadence', np.median(np.diff(nights)))


"""
import matplotlib.pyplot as plt
plt.plot(allobs['RA'], allobs['Dec'], 'ko')
plt.show()
"""
