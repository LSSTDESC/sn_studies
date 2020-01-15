from sn_tools.sn_io import getObservations
from sn_tools.sn_cadence_tools import ClusterObs
from sn_tools.sn_obs import renameFields,getFields,season

import os
import numpy as np
from . import plt
import pandas as pd

class DDCluster:
    def __init__(self,dbDir, dbName,dbExtens,nclusters):
        """
        class to make a set of clusters in (RA,Dec)
        out of a set of simulated data

        Parameters
        ----------
        dbDir: str
         path to the location dir of the database
        dbName: str
         name of the dbfile to load
        dbExtens: str
         extension of the dbfile
        two possibilities:
        - dbExtens = db for scheduler files
        - dbExtens = npy for npy files (generated from scheduler files)

        """

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
       

        # load the data
        data = self.dataDB()

        #remove the u-band
        idx = data['filter'] != 'u'

        data = data[idx]

        # make clusters of data
        self.dataclusters = self.clusterDD(nclusters,data)


    def dataDB(self):
        """Function to load observations
        from a simulation db

        Parameters
        ----------

        Returns
        ------
        numpy record array with scheduler information like
        observationId, fieldRA, fieldDec, observationStartMJD, 
        flush_by_mjd, visitExposureTime, filter, rotSkyPos, 
        numExposures, airmass, seeingFwhm500, seeingFwhmEff, 
        seeingFwhmGeom, sky, night, slewTime, visitTime, 
        slewDistance, fiveSigmaDepth, altitude, azimuth, 
        paraAngle, cloud, moonAlt, sunAlt, note, fieldId,
        proposalId, block_id, observationStartLST, rotTelPos, 
        moonAz, sunAz, sunRA, sunDec, moonRA, moonDec, 
        moonDistance, solarElong, moonPhase
        This list may change from file to file.
        """

        # loading observations

        observations = getObservations(self.dbDir, self.dbName,self.dbExtens)
    
        #rename fields

        observations = renameFields(observations)

        #print(observations.dtype.names)

        return observations


    def clusterDD(self,nclusters,observations):
        """ 
        Function to identify clusters of points (Ra,Dec)
        for a set of observations.

        Parameters
        ----------
        nclusters: int
         number of clusters to find
        observations: numpy record array
         array of observations

        """

        # get DD observations only
        fieldIds = [290,744,1427, 2412, 2786]
        
        observations_DD = getFields(observations,'DD', fieldIds,128)
        # get clusters out of these obs

        dataclusters = ClusterObs(observations_DD,nclusters=nclusters,dbName=self.dbName).dataclus
        return dataclusters

def dataWrapper(dbDir,dbName,dbExtens,nclusters):
    """
    Function to get data used for analysis

    Parameters
    ----------
    dbDir: str
     path to the location dir of the database
    dbName: str
     name of the dbfile to load
    dbExtens: str
     extension of the dbfile
     two possibilities:
     - dbExtens = db for scheduler files
     - dbExtens = npy for npy files (generated from scheduler files)
    nclusters: int
     number of clusters to consider

    Returns
    -------
    pandas df with original data + cluster infos + season info

    """

    dataName ='Clusters_{}.npy'.format(dbName)

    if not os.path.exists(dataName):
        clusters = DDCluster(dbDir,dbName,dbExtens,nclusters).dataclusters
        np.save(dataName,np.copy(clusters.to_records(index=False)))

    # load data
    dataclusters = np.load(dataName)

    #estimate seasons

    restot = None
    for fieldName in np.unique(dataclusters['fieldName']):
        idx = dataclusters['fieldName']==fieldName
        sel = dataclusters[idx]
        if restot is None:
            restot = season(sel)
        else:
            restot = np.concatenate([restot,season(sel)])

   
    return pd.DataFrame(np.copy(restot))
    
class StatSim:
    def __init__(self,dbDir,dbName,dbExtens,nclusters):
        """
        class to estimate some stats (such as then median number of visits, the season length, ...)

        Parameters
        ----------
        dbDir: str
         path to the location dir of the database
        dbName: str
         name of the dbfile to load
        dbExtens: str
         extension of the dbfile
         two possibilities:
         - dbExtens = db for scheduler files
         - dbExtens = npy for npy files (generated from scheduler files)
        nclusters: int
         number of clusters to consider
        """

        self.dbName = dbName
        #grab the data
        datadf = dataWrapper(dbDir,dbName,dbExtens,nclusters)

        # summary per night
        fName = 'Summary_night_{}.npy'.format(dbName)

        if not os.path.exists(fName):
            grp_filter = datadf.groupby(['fieldName','filter','night'])
            summary = grp_filter.apply(lambda x: self.summarynight(x)).reset_index()
            np.save(fName,summary.to_records())

        summary = pd.DataFrame(np.load(fName))

        self.mednight = summary

        
        sumhori = summary.groupby(['fieldName','night']).apply(lambda x: self.horizont(x)).reset_index()

        #print(sumhori['fieldName'].unique())
        #idf = sumhori['fieldName'] == 'COSMOS'
        #sumhori = sumhori[idf]


        #get the numExposures median values
        varmeds = ['numExposures_g','numExposures_r',
                   'numExposures_i','numExposures_z',
                   'numExposures_y']

        #for vv in varmeds:
        #    sumhori[vv] = sumhori[vv].replace(0, np.NaN)

        medians = sumhori.groupby(['fieldName','season'])[varmeds].median().reset_index()
        
        
        for vv in varmeds:
            sumhori[vv] = sumhori[vv].replace(np.NaN,0)
        
        for b in 'grizy':
            medians['numExposures_{}'.format(b)] = medians['numExposures_{}'.format(b)].astype(int)


        medians['numExposures'] = medians[['numExposures_g','numExposures_r',
                                   'numExposures_i','numExposures_z',
                                   'numExposures_y']].sum(axis=1)
        

        #medians = medians.rename(columns={"numExposures": "numExposures_med"})


        # merge sumhori with medians
        #sumhori = sumhori.merge(medians,left_on=['clusId','fieldName'],right_on=['clusId','fieldName'])

        # final analysis: cadence and season_length estimations
        dffi = sumhori.groupby(['fieldName','season']).apply(lambda x: self.anaTemp(x)).reset_index()

        # merge with medians
        self.data=dffi.merge(medians,left_on=['fieldName','season'],right_on=['fieldName','season'])


    def summarynight(self,grp):
        """
        Method to perform some calc on a pandas grp

        Parameters
        ----------
        grp: pandas df
         data to process

        Returns
        -------
        pandas df with the following cols:
         visitExposureTime,numExposures: sum
         observationStartMJD,season,airmass, 
         seeingFwhm500, seeingFwhmEff,
         seeingFwhmGeom, sky,
         moonRA, moonDec, moonDistance, 
         solarElong,moonPhase, 
         healpixID, pixRa, pixDec, 
         ebv, RA, Dec, fiveSigmaDepth_med: median
         fiveSigmaDepth: m5 combined (+1.25*log(Nvisits))
        """


        var_sum = ['visitExposureTime','numExposures']
        var_median = ['observationStartMJD','season','airmass', 
                      'seeingFwhm500', 'seeingFwhmEff',
                      'seeingFwhmGeom', 'sky',
                      'moonRA', 'moonDec', 'moonDistance', 
                      'solarElong','moonPhase', 
                      'healpixID', 'pixRa', 'pixDec', 
                      'ebv', 'RA', 'Dec','fiveSigmaDepth',
                      'fieldRA','fieldDec','proposalId','fieldId']
        
        dfout = pd.DataFrame()

        for v in var_sum:
            dfout[v]=[grp[v].sum()]
    
        for v in var_median:
            dfout[v]=[grp[v].median()]

        dfout = dfout.rename(columns={'fiveSigmaDepth':'fiveSigmaDepth_med'})
        #print(grp['fiveSigmaDepth'],np.log10(dfout['numExposures']))
        dfout['fiveSigmaDepth']= dfout['fiveSigmaDepth_med']+1.25*np.log10(dfout['numExposures'])
        

        return dfout

    def horizont(self,grp):
        """
        Method to transform grp info (cols) to row df
        
        Parameters
        ---------
        grp: pandas df
         data to process
    
        Returns
        -------
        pandas df with the following cols:
        numExposures_g,r,i,z,y: number of visits per band
        observationStartMJD,season: median values
        
        """

        dvisits = {} 
        var= 'numExposures'
        for b in 'grizy':
            io = grp['filter'] == b
            dvisits['{}_{}'.format(var,b)] = [grp[io][var].median()]

        dfout = pd.DataFrame(dvisits)
        dfout = dfout.fillna(0.)
        for b in 'grizy':
            dfout['{}_{}'.format(var,b)] = dfout['{}_{}'.format(var,b)].astype(int)

        for var in ['observationStartMJD','season']:
            dfout[var] =  grp[var].median()

        return dfout

    def anaTemp(self,grp):
        """
        Method that estimates the cadence and the season length of a group of data

        Parameters
        ----------
        grp: pandas df
         data to process
        
        Returns
        -------
        pandas df with season_length and cadence as cols.

        """

        grp = grp.sort_values(by=['observationStartMJD'])
        season_length = grp['observationStartMJD'].max()-grp['observationStartMJD'].min()
        cadence = grp['observationStartMJD'].diff().iloc[1:].median()


        return pd.DataFrame({'season_length':[season_length],
                             'cadence':[cadence]})


    def plot(self, var_to_plot,ylabel):
        """
        Method to plot data

        Parameters
        ----------
        var_to_plot: str
         variable to plot
        ylabel: str
         variable label (y-axis)

        """

        if var_to_plot not in self.data.columns:
            print('The variable you are trying to plot does not exist')
            print('You should choose among :',self.data.columns)



        colors=dict(zip(range(1,13),['red','blue','orange','black',
                                    'green','pink','brown','purple',
                                    'pink','magenta','olive','cyan']))

        fig, ax = plt.subplots()
        fig.suptitle(self.dbName)
        for seas in self.data['season'].unique():
            idx = self.data['season']==seas
            sel = self.data[idx]
            ax.plot(sel['fieldName'],sel[var_to_plot],
                    label='season {}'.format(int(seas)),color=colors[seas])
            
        ax.set_xlabel(r'Field')
        ax.set_ylabel(r'{}'.format(ylabel))
        ax.legend(ncol=6, loc='best')



def duplicate(grp,nvisits_night):

    grp = grp.sort_values(by=['observationStartMJD'])
    max_mjd = grp['observationStartMJD'].max()

    df_night = pd.DataFrame()
    time_shift = 0.
    time_slew = 60.
    for b in nvisits_night.keys():
        dfb = pd.DataFrame(list(range(nvisits_night[b])),columns=['delta_mjd'])
        dfb['delta_mjd'] = 34.*dfb['delta_mjd']+time_shift
        dfb.loc[:,'filter']=b
        time_shift = dfb['delta_mjd'].max()+time_slew
        dfb['delta_mjd'] = dfb['delta_mjd']/(24.*3600.)
        df_night = pd.concat([df_night, dfb], sort=False)

    #print(df_night)

    grp_new = grp.merge(df_night,left_on=['filter'],right_on=['filter'])

    grp_new['observationStartMJD'] = grp_new['observationStartMJD']+grp_new['delta_mjd']
    grp_new = grp_new.drop(columns=['night'])

    #print(grp_new.columns)

    idx = grp_new['observationStartMJD']<=max_mjd

    return grp_new
    #return grp_new[idx]
