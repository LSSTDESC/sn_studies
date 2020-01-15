from scipy import interpolate
import pandas as pd
import numpy as np


class ObsReshuffled:
    def __init__(self,data, datasummary, visits_ref, zlim, dbName):
        """
        class to make Reshuffled observations
        The idea is to estimate the number of visits(per band) 
        requested to reach zlim and to build a set of observations 
        with these numbers by modifying the data 
        (mainly the columns numExposures, visitExposureTime, fiveSigmaDepth)  
        
        Parameters
        ----------
        data: pandas df
         data to reshuffle
        datasummary: pandas df
         df with a summary of the data (i.e. some stat per season such as the season_length, or the cadence)
        visits_ref: numpy array
         array with the number of visits (per band) for a set of cadences
        zlim: float
         redshift value used to estimate the requested number of visits

        """

        # select vals corresponding to zlim
        idx = np.abs(visits_ref['z']-zlim)<1.e-5
        sel_visits = visits_ref[idx]
        
        self.dbName = dbName
        self.zlim = zlim
        self.makeObs(data,datasummary,sel_visits)


    def makeObs(self,data,datasummary,visits):
        """
        Method to reshuffle data
        Parameters
        ----------
        data: pandas df
         data to reshuffle
        visits: numpy array
         array with the number of visits (per band) for a set of cadences
        
        Returns
        -------
         panda df stored as a npy file


        """

        # select only some of the cols
        extr = datasummary[['fieldName', 'season', 'season_length', 'cadence']].copy()

        resdf = pd.DataFrame()
        for band in 'rizy':
            add = self.nVisits(extr,visits,band)
            resdf = pd.concat([resdf,add],sort=False)
        ida = resdf['Nvisits']>0
        print('hello',resdf[ida])
        assoc = data.merge(resdf,left_on=['fieldName', 'season','filter'],right_on=['fieldName', 'season','filter'])

        # Make some modifs here to integrate the number of visits
        assoc.loc[:,'numExposures'] = assoc['Nvisits']
        assoc['visitExposureTime'] = assoc['numExposures']*30
        assoc['fiveSigmaDepth'] = assoc['fiveSigmaDepth_med']+1.25*np.log10(assoc['numExposures'])

        idx = assoc['numExposures']>0
        assoc = assoc[idx]

        print('assoc obs',assoc[['fieldName', 'season','filter','numExposures','visitExposureTime','fiveSigmaDepth','fiveSigmaDepth_med','Nvisits']])
        # drop unwanted columns
        todrop = ['Nvisits','cadence','season_length','fieldName',
                  'season','RA','Dec','healpixID','pixRa','pixDec','ebv']

        assoc = assoc.drop(columns=todrop)
        assoc['proposalId'] = assoc['proposalId'].astype(int)
        assoc['fieldId'] = assoc['fieldId'].astype(int)


        
        outName = '{}_{}.npy'.format(self.dbName,np.around(self.zlim,decimals=2))

        np.save(outName,np.copy(assoc[idx].to_records(index=False)))


    def nVisits(self,grp,ref,b):
        """
        Method to estimate the number of visits as a function of the cadence

        Parameters
        ----------
        grp: pandas df
         pandas df used to estimate the number of visits. Should at least have a cadence column
        ref: numpy array
         array with at least the cadence and Nvisits_band cols
        b: str
         band to consider

        Returns
        -------
        grp: pandas df
         original df with two additional cols: filter and Nvisits

        """
        
        resdf = grp.copy()
        # prepare to compute with interp1d
        interp = interpolate.interp1d(ref['cadence'],ref['Nvisits_{}'.format(b)],
                                      bounds_error=False,fill_value=0.0)
        #now append two cols on the pandas df: filter and Nvisits
        resdf.loc[:,'filter'] = b
        resdf.loc[:,'Nvisits'.format(b)] = np.round(interp(grp['cadence'])).astype(int)


        return resdf
