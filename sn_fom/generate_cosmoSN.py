from optparse import OptionParser
import numpy as np
import re

def append(fields,nultra=0):
    """
    Function to append a field

    Parameters
    ----------
    fields: list(str)
      list of the fields to append
    nultra: int, opt
      number of ultradeep fields

    Returns
    -------
    the line 

    """
    lia = ''
    for field in fields[:-1]:
        lia += '{},'.format(field)
    lia += '{};'.format(fields[-1])
    
    if nultra >= 1 and len(fields)>nultra:
        r = []
        for match in re.finditer(',',lia):
            r.append(match.start())
        n = r[nultra-1]
        lia = lia[0:n] + '/' + lia[n+1: ]
    return lia

def add_config(fi, prefix,nultra,fields,
               nseasons,pointings,runtype,
               iconf,suffix):
    """
    Method to add a configuration

    Parameters
    ----------
    fi: 
      the file where to write the results
    prefix: str
      prefix for the first field/usually used when nultra>0
    nultra: int
      number of ultradeep fields
    ntot: int
      total number of fields
    fields: list(str)
      list of fields to consider
    nseasons: list(str)
      list of seasons to consider
    pointings: list(int)
      list of pointings to consider
    runtype: str
      type of run (e.g. universale, deep_rolling, ...)
    iconf: int
      tag for configuration



    """
    zcomp = np.arange(0.55,0.95,0.05)
    nfields = np.sum(pointings)
    print('# {} fields'.format(nfields))
    fi.write('# {} fields \n'.format(nfields))
    for z in zcomp:
        iconf += 1
        li = prefix
        if len(fields) > nultra:
            if nultra > 0:
                li += '/'
            li += 'DD_{}'.format(np.round(z,2)).ljust(7,'0')
        li += ';'
        li += append(fields,nultra)
        li += append(nseasons,nultra)
        li += append(pointings,nultra)
        li += 'conf_{}_{}_{}'.format(iconf,runtype,suffix)
        fi.write(li+'\n')
        print(li)
        if nfields == nultra or nfields==1:
            break
        

    return iconf

parser = OptionParser()

parser.add_option('--ultraDeep', type=str, default='COSMOS,XMM-LSS',
                  help='list of ultra deep fields[%default]')
parser.add_option('--z_ultra', type=str, default='0.90,0.90',
                  help='z complete of ultra deep fields [%default]')
parser.add_option('--nseason_ultra', type=str, default='2,2',
                  help='number of visits for ultra deep fields[%default]')
parser.add_option('--Deep', type=str, default='CDFS,ELAIS,ADFS',
                  help='list of ultra deep fields[%default]')
parser.add_option('--nseason_Deep', type=str, default='2,2,2',
                  help='number of visits for deep fields[%default]')


opts, args = parser.parse_args()

ultraDeepFields = opts.ultraDeep.split(',')
z_ultra = opts.z_ultra.split(',')
nseasons_ultra = opts.nseason_ultra.split(',')
nultra = len(ultraDeepFields)

DeepFields = opts.Deep.split(',')
nseasons_Deep= list(map(int, opts.nseason_Deep.split(',')))

    
print(ultraDeepFields,DeepFields,nseasons_Deep,nseasons_ultra)

fields = ultraDeepFields+DeepFields
fields = list(filter(None,fields))
nseasons = nseasons_ultra+nseasons_Deep
nseasons = list(filter(None,nseasons))


run_type = 'deep_rolling'
prefix = 'DD_{}'.format(np.unique(z_ultra)[0])
suffix = ''
if ultraDeepFields == ['']:
    run_type = 'universal'
    prefix = ''
    suffix = np.unique(list(map(int,nseasons)))[0]
    nultra = 0
else:
    suffix = '_'.join(z_ultra)+'_'+'_'.join(nseasons_ultra)

cvsName = 'config_cosmoSN_{}_{}.csv'.format(run_type,suffix)

fi = open(cvsName,'w')
bandeau = 'dbName;fields;nseasons;npointings;configName'
fi.write(bandeau +'\n')

fields_pointings = ['COSMOS','XMM-LSS','CDFS','ELAIS','ADFS']
pointings_max = dict(zip(fields_pointings,[1,1,1,1,2]))
pointings_uni = dict(zip(fields_pointings,[1,1,1,1,1]))

configs = {}

configs[0] = {}
configs[0]['fields'] = fields
configs[0]['nseasons'] = nseasons
pp = []
for fil in fields:
    pp.append(pointings_max[fil])
configs[0]['npointings'] = pp

configs[1] = {}
configs[1]['fields'] = fields
configs[1]['nseasons'] = nseasons
pp_uni = []
for fil in fields:
    pp_uni.append(pointings_uni[fil])
configs[1]['npointings'] = pp_uni

for i in range(len(fields)-1):
    configs[i+2] = {}
    configs[i+2]['fields'] = fields[:-i-1]
    configs[i+2]['nseasons'] = nseasons[:-i-1]
    configs[i+2]['npointings'] = pp_uni[:-i-1]


iconf = 0
print('# {} survey'.format(run_type))
fi.write('# {} survey \n'.format(run_type))
for key, vals in configs.items():
    iconf = add_config(fi, prefix, nultra, 
                       vals['fields'],vals['nseasons'],
                       vals['npointings'],
                       run_type,iconf,suffix)
    
fi.close()
    
