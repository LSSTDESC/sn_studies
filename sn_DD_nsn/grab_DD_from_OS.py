import numpy as np
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_tools.sn_obs import DDFields
from sn_tools.sn_clusters import ClusterObs
from sn_tools.sn_obs import season

class TransformDDF:

    def __init__(self,dbDir,dbName,dbExtens,fieldList,DDF,nclusters=6):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
        fields=['COSMOS','CDFS','ELAIS','XMM-LSS','ADFS1','ADFS2']
        colors = 'rgbkym'
        colordisp = dict(zip(fields,colors))

        # load data
        res = pd.DataFrame(self.loadFile())

        # select DDF from note
        
        idx = res['note'].isin(fieldList)
        resb = res[idx]
        #these two fields are requested for the clustering
        resb.loc[:,'fieldRA'] = resb['RA']
        resb.loc[:,'fieldDec'] = resb['Dec']

        #make clusters
        clusters = ClusterObs(resb.to_records(index=False), nclusters, dbName, DDF).dataclus
        # get seasons
        print('hello',clusters.columns)
        clusters = season(clusters.to_records(),mjdCol='mjd')
        # plot clusters
        #self.plot(clusters,colordisp)

        #remove dithering
        clus_no_dither = pd.DataFrame(np.copy(clusters)).groupby(['fieldName']).apply(lambda x:self.removeDithering(x)).reset_index()
        
        colors = 'kkkkkk'
        colordisp = dict(zip(fields,colors))
        #self.plot(clus_no_dither,colordisp)
        
        #set m5 to median m5 per season and per filter
        clus_median_m5 = clus_no_dither.groupby(['fieldName','season','band']).apply(lambda x:self.m5_median_season(x))

        idx = clus_median_m5['fieldName']=='COSMOS'
        sel = clus_median_m5[idx]
        for band in np.unique(sel['band']):
            fig, ax = plt.subplots()
            idxb = sel['band']==band
            selb = sel[idxb]
            print('ahahah',np.unique(selb['band']))
            ax.plot(selb['mjd'],selb['fiveSigmaDepth'],'ko')
        plt.show()
        #complete missing obs
        
        
        plt.show()

    def loadFile(self):

        fullName = '{}/{}.{}'.format(self.dbDir,self.dbName,self.dbExtens)

        res = np.load(fullName)

        return res

    def removeDithering(self, grp):
        
        grp['fieldRA'] = grp['fieldRA'].mean()
        grp['fieldDec'] = grp['fieldDec'].mean()

        return grp

    def m5_median_season(self,grp):

        grp['fiveSigmaDepth'] = grp['fiveSigmaDepth'].median()

        return grp


    def plot(self,clusters,colordisp):
        
        for fieldName in np.unique(clusters['fieldName']):
            idx = clusters['fieldName']==fieldName
            sel = clusters[idx]
            plt.plot(sel['fieldRA'],sel['fieldDec'],marker='o',color=colordisp[fieldName],ls='None')

        
        

parser = OptionParser()

parser.add_option('--dbList', type='str', default='for_batch/input/List_Db_DD.csv',help='list of dbNames to process  [%default]')


opts, args = parser.parse_args()

toprocess = pd.read_csv(opts.dbList, comment='#')

fieldList = ['DD:COSMOS','DD:ECDFS','DD:EDFS','DD:ELAISS1','DD:XMM-LSS']
fields=['COSMOS','CDFS','ELAIS','XMM-LSS','ADFS1','ADFS2']
colors = 'rgbkym'
colordisp = dict(zip(fields,colors))

DDF = DDFields()
print(DDF)
nclusters = 6
for i,row in toprocess.iterrows():

    TransformDDF(row['dbDir'],row['dbName'],row['dbExtens'],fieldList,DDF)
    

    break
