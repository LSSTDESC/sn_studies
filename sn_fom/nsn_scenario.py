import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rf


class Mod_z:
    """
    Method to modify values of input file

    """

    def __init__(self, fName):

        # load the file

        tab = np.load(fName, allow_pickle=True)

        tabdf = pd.DataFrame.from_records(tab)

        tabdf = tabdf.replace('ADFS1', 'ADFS')

        tabmod = tabdf.groupby(['cadence']).apply(
            lambda x: self.mod(x))

        # tabmod = tabdf
        ll = []
        for b in 'grizy':
            ll.append('Nvisits_{}'.format(b))
        tabmod['Nvisits'] = tabmod[ll].sum(axis=1)

        self.nvisits = tabmod

    def mod(self, grp):
        """
        Method to modify a group


        Parameters
        --------------
        grp : pandas df group

        Returns
        -----------
        modified grp group
        """
        for band in 'r':
            what = 'Nvisits_{}'.format(band)
            idx = grp[what] > 1.e-21
            idx &= grp[what] < 1.
            grp.loc[idx, what] = 1.

            if band == 'g' or band == 'r':
                Index_label = grp[grp[what] < 1.e-10].index.tolist()
                Index_label_p = grp[grp[what] > 1.e-10].index.tolist()
                # print('index', Index_label, Index_label_p)
                grp.loc[Index_label, what] = grp.loc[Index_label_p[-1]][what]
            # print('io', grp[what])

        grp['Nvisits_g'] = 2
        # grp['Nvisits_r'] = 4
        return grp


class SeasonLength:
    """"
    class to estimate SN season length depending on survey params
    (zcomplete)

    Parameters
    ---------------
    fDir: str
      location dir of the Nvisits vs zcomplete file
    fName: str
      name of the Nvisits vs zcomplete file
    slDir: str
       location dir of the season length vs exptime file
    slName: str
       name of the season length vs exptime file
    """

    def __init__(self, fDir, fName, slDir, slName):

        # loading Nvisits vs zcomplete
        self.zlim_visit, self.visit_zlim = self.load(fDir, fName)

        # loading season length vs exptime
        self.nvisits_seasonlength = self.load_seasonlength_nvisits(
            slDir, slName)

    def load(self, fDir, fName):
        """
        Method to load a file fDir/fName

        Parameters
        --------------
        fDir: str
         location dir of the file
        fName: str
         file name

        Returns
        -----------
        dict of interp1d
        key: cadence
          values: interp1d(nvisits, z)

        """
        # tab = np.load(fName, allow_pickle=True)
        tab = Mod_z('{}/{}'.format(fDir, fName)).nvisits
        res = {}
        resb = {}
        print(tab)
        for cad in np.unique(tab['cadence']):
            idx = tab['cadence'] == cad
            sel = tab[idx]
            res[cad] = interp1d(sel['Nvisits'], sel['z'],
                                bounds_error=False, fill_value=0.)
            resb[cad] = interp1d(sel['z'], sel['Nvisits'],
                                 bounds_error=False, fill_value=0.)
        return res, resb

    def load_seasonlength_nvisits(self, slDir, slName):
        """
        Method to load (seasonlength, nvisits) file and make interp1d out of it

        Parameters
        ---------------
        slDir: str
          location dir of (seasonlength vs nvisits) file
        slName: str
          (seasonlength vs nvisits)

        Returns
        -----------
        dict of interp1d(nvisits, seasonlength); key: field name
        """

        tab = np.load('{}/{}'.format(slDir, slName), allow_pickle=True)

        res = {}
        self.DD_list = np.unique(tab['name']).tolist()

        for fieldName in np.unique(tab['name']):
            idx = tab['name'] == fieldName
            sel = tab[idx]
            res[fieldName] = interp1d(sel['nvisits'], sel['season_length'],
                                      bounds_error=False, fill_value=0.)
        return res

    def __call__(self, config, cadence=1):
        """
        Method to estimate the seasone lengths

        Parameters
        --------------
        config: numpy array
        cadence: int, opt
          cadence of observation (default: 1 day)

        Returns
        ----------


        """
        r = []
        for fieldName in config['fieldName']:
            idx = config['fieldName'] == fieldName
            sel = config[idx]
            zcomp = sel['zcomp'].item()
            max_season_length = sel['max_season_length'].item()
            # get the number of visits
            nvisits = self.visit_zlim[cadence](zcomp).item()
            season_length = self.nvisits_seasonlength[fieldName](nvisits)
            season_length = np.min([season_length, max_season_length])
            r.append(np.round(season_length, 1))

        res = rf.append_fields(config, 'season_length', r)

        return res


class NSN_scenario:
    """
    class to estimate the number of SN

    """

    def __init__(self):

        from sn_tools.sn_rate import SN_Rate

        self.rateSN = SN_Rate(H0=70., Om0=0.3,
                              min_rf_phase=-15., max_rf_phase=30)

    def __call__(self, config):

        zmin = 0.01
        zstep = 1.e-3

        res = None
        for fieldName in config['fieldName']:
            idx = config['fieldName'] == fieldName
            sel = config[idx]
            tb = self.calc_nsn(sel)
            tb = pd.DataFrame(np.copy(tb))
            if res is None:
                res = tb
            else:
                res = pd.concat((res, tb))

        return res

    def calc_nsn(self, sel):

        fieldName = sel['fieldName'].item()
        zcomp = sel['zcomp'].item()
        season_length = sel['season_length'].item()
        area = sel['survey_area'].item()
        nseasons = sel['nseasons'].item()
        nfields = sel['nfields'].item()

        print(fieldName, zcomp, season_length, area, nseasons, nfields)

        zmin = 0.05
        zstep = 0.05

        zvals = np.arange(zmin, zcomp+zstep, zstep).tolist()

        r = [(0.01, 0.)]
        for z in zvals:
            zzmax = np.round(z, 2)
            zz, rate, err_rate, nsn, err_nsn = self.rateSN(zmin=0.01,
                                                           zmax=zzmax,
                                                           dz=0.001,
                                                           duration=season_length,
                                                           survey_area=area,
                                                           account_for_edges=True)
            nsn = np.round(np.cumsum(nsn)[-1], 4)
            r.append((zzmax, nsn))

        res = np.rec.fromrecords(r, names=['z', 'nsn_season'])

        res = rf.append_fields(res, 'fieldName', [fieldName]*len(res))
        res = rf.append_fields(res, 'zcomp', [zcomp]*len(res))
        res = rf.append_fields(res, 'season_length', [season_length]*len(res))
        res = rf.append_fields(res, 'survey_area', [area]*len(res))
        res = rf.append_fields(res, 'nseasons', [nseasons]*len(res))
        res = rf.append_fields(res, 'nfields', [nfields]*len(res))

        return res


class NSN_config:

    def __init__(self, config):
        # get season lengths
        fDir = 'sn_studies/input'
        fName = 'Nvisits_z_-2.0_0.2_error_model_ebvofMW_0.0_nvisits_Ny_20.npy'
        slDir = fDir
        slName = 'seasonlength_nvisits.npy'
        sl = SeasonLength(fDir, fName, slDir, slName)
        config = sl(config)

        print('here sl', config)
        # get nsn
        nsn = NSN_scenario()
        config = nsn(config)

        config['nsn_survey'] = config['nseasons'] * \
            config['nfields']*config['nsn_season']

        self.data = config

    def nsn_tot(self):

        idx = np.abs(self.data['z']-self.data['zcomp']) < 1.e-5

        return np.sum(self.data[idx]['nsn_survey'])


"""
r = []
r.append(('COSMOS', 0.9, 180., 1, 9.6, 2))
r.append(('XMM-LSS', 0.9, 180., 1, 9.6, 2))
r.append(('ELAIS', 0.65, 180., 1, 9.6, 2))
r.append(('CDFS', 0.65, 180., 1, 9.6, 2))
r.append(('ADFS', 0.65, 180., 2, 9.6, 2))

config = np.rec.fromrecords(
    r, names=['fieldName', 'zcomp', 'max_season_length', 'nfields', 'survey_area', 'nseasons'])


res = nsn_config(config)

print(res)
"""
