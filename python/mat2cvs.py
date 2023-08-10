from pyspark import SparkContext

sc = SparkContext(master=master_url)

get_ipython().system('pip install pandas')

get_ipython().system('pip install scipy')

from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

get_ipython().magic('pylab inline')

import pandas as pd
import datetime as dt

from scipy.io import loadmat,savemat,whosmat

from string import split
from collections import Counter
import re
import numpy as np
from numpy import shape

from glob import glob


get_ipython().magic('cd /root/ipython/BeakedWhaleClassification/')
get_ipython().magic('run Credentials.ipynb')

s3helper.set_credential(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

s3helper.open_bucket('while-classification')

dirs=[str(a) for a in s3helper.ls_s3('') if '_' in a]
dirs

# check avaiable local space
get_ipython().system('df')

#%%writefile matlab2datenum.py
def matlab2datetime(matlab_datenum):
    try:
        day = dt.datetime.fromordinal(int(matlab_datenum))
        dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
        return day + dayfrac
    except:
        return None

def mat2columns(data,label):
    """ load a 2d number array as columns in D 
            data: 2d numpy array
            label: the prefix of the columns (followed by index)
            D: assumed to be an existing dataframe, defined externally.
    """
    for i in range(shape(data)[1]):
        col=label+str(i)
        D[col]=data[:,i]
        columns.append(col)

def mat2packedarrays(data):
    """ represent a 2D array as a list of binary arrays"""
    m,n = data.shape
    L=[]
    for i in range(m):
        L.append(packArray(data[i,:],))
    return L


# %load numpy_pack.py
import numpy as np
"""Code for packing and unpacking a numpy array into a byte array.
   the array is flattened if it is not 1D.
   This is intended to be used as the interface for storing 
   
   This code is intended to be used to store numpy array as fields in a dataframe and then store the 
   dataframes in a parquet file.
"""

def packArray(a):
    if type(a)!=np.ndarray:
        raise Exception("input to packArray should be numpy.ndarray. It is instead "+str(type(a)))
    return bytearray(a.tobytes())
def unpackArray(x,data_type=np.int16):
    return np.frombuffer(x,dtype=data_type)

pattern=re.compile(r'(\w{2})(\d{2})_([^_]+)_([A-Z]+)(\d*).mat')
def parse_filename(filename):
    match=pattern.search(filename)
    if match:
        site,rec_no,species,fields,no=match.groups()
        if no=='':  # if no iteration number, call it iteration 1.
            no='1'
        return (site,rec_no,species,fields,no)
    else:
        return -1
parse_filename('DT_Cuviers/DT02_Cuviers_FD.mat')

def row2cvs(T,lrow,fmt='%6.4f',date_format='%Y-%m-%d %H:%M:%S.%f'):
    L=[dt.datetime.strftime(T,date_format)]
    for x in lrow:
        #print type(x)
        if type(x)==type('string'):
            L.append(x)
        elif type(x)==np.float64:
            L.append(fmt%x)
        elif type(x)==np.bool_:
            L.append('%1d'%(1*x))
        elif type(x)==numpy.int64:
            L.append(str(x))
        elif type(x)==bytearray:
            L.append(packed2cvs(x))
        else:
            raise Exception('row2cvs error: unrecognized type='+str(type(x)))
    return ','.join(L)+'\n'

def packed2cvs(row, data_type=np.float64,fmt='%6.4f'):
    return ','.join([fmt%x for x in list(unpackArray(row,data_type=data_type))])

#Copy mat files to local directory
pattern=re.compile(r'(\w{2})(\d{2})_([^_]+)_([A-Z]+)(\d*).mat')
for _dir in dirs:
    get_ipython().system('mkdir /mnt/whales/$_dir')
    List=s3helper.ls_s3(_dir)
    master_key=0
    for _file in List:
        print _dir,_file
        key=parse_filename(_file)
        if key==-1:
            continue
        print key
        s3helper.s3_to_local(_file,'/mnt/whales/'+_file)
#        if master_key==0:
#            master_key=key[:3]
#            print master_key
#        else:
#            if key[:3] != master_key:
                

get_ipython().system('ls -l /mnt/whales/* | wc')
get_ipython().system('df')
get_ipython().system('du -s -h /mnt/whales/*')
#s3helper.s3_to_local('DT_Cuviers/DT04_Cuviers_TPWS2.mat','/mnt/whales/DT_Cuviers/DT04_Cuviers_TPWS2.mat')



from glob import glob
get_ipython().magic('cd "/mnt/whales"')
get_ipython().system('mkdir /mnt/whales_CVS')

dirs=glob('/mnt/whales/*')
print dirs
for _dir in dirs:
    print _dir
    print '\n','='*50
    List=glob(_dir+"/*.mat")
    Keys = [parse_filename(a) for a in List]
    Master_Keys=list(set([a[:3] for a in Keys]))
    #print 'master keys for ',_dir,'\n',Master_Keys

    # read mat files for one recordings
    for _master in Master_Keys:
        print '\nprocessing','.'.join(_master)
        print '-'*50
        data={}
        for filename in List:
            key=parse_filename(filename)
            if key[:3]!=_master:
                continue
            print 'loading %s into %s'%(filename,key)
            data[key]=loadmat(filename)
            #print '====',key,':'

            for key2 in data[key].keys():
                if key2[:2]=='__':
                    del data[key][key2]
                else:
                    pass
                    #print key2, shape(data[key][key2])
                    
        print '\n read TPWS files into a pandas dataframe'
        
        mdf=None  # mdf is the master data frame into which all of the data is collected.

        # load TPWS files
        for file_key in [k for k in data.keys() if k[:4]==_master[:3]+('TPWS',)]:
            #print file_key

            TPWS=data[file_key]

            D={}
            columns=['time','species','site','rec_no','bout_i','peak2peak','MSN','MSP']
            D['time']=[matlab2datetime(t) for t in TPWS['MTT'][0,:]]
            D['site']=_master[0]
            D['rec_no']=_master[1]
            D['species']=_master[2]
            D['peak2peak']=TPWS['MPP'][0,:]

            D['MSN'] = mat2packedarrays(TPWS['MSN'])
            D['MSP'] = mat2packedarrays(TPWS['MSP'])

            df=pd.DataFrame(D,columns=columns)
            df.index=D['time']
            df=df[columns[1:]]
            if type(mdf)==type(None):
                mdf=df
            else:
                # add rows that do not currently exist in mdf
                df_new=df.select(lambda x: not x in mdf.index)
                mdf=pd.concat([mdf,df_new])
            print 'after adding %s%s'%file_key[-2:],df.shape,mdf.shape
        
        print "\n merge all data items (.mat files) corresponding to site, rec_no and species"

        akeys=[]
        
        #add indicator columns.
        for col in ['TPWS1','MD1','FD1','TPWS2','MD2','FD2','TPWS3','MD3','FD3']:
            mdf[col]=False
        
        for file_key in [k for k in data.keys() if k[:3]==_master[:3]]:

            tbl=data[file_key]
            array_keys=tbl.keys()
            if len(array_keys)==1:
                time_key=array_keys[0]
            else:
                time_key='MTT'


            D2=[matlab2datetime(t) for t in np.ravel(tbl[time_key])]


            if len(D2)==0:
                print file_key,'is empty'
                continue

            S3=pd.Series(data=True,index=D2)
            # Remove from S3 entries with a bad index
            nans=[(a!= None and type(a)!=pd.tslib.NaTType) for a in S3.index]
            S3=S3[nans]
            mdf_key=file_key[-2]+file_key[-1]
            akeys.append(mdf_key)
            if not mdf_key in mdf.columns:
                print 'ERROR: the key %s is not predifined'%mdf_key
            else:
                mdf[mdf_key]=S3
        mdf=mdf.fillna(False)
        print 'after adding all',mdf.shape
        print 'akeys=',akeys
        
        
        print '\n find time gaps (breaks) larger the 1800 seconds = 30 minutes',
        #these time gaps define what is a recording from a single bout of whales.

        mdf=mdf.sort_index()
        times=mdf.index
        deltas=np.array([(times[i+1]-times[i]).total_seconds() for i in range(len(times)-1)])
        bout_i = np.concatenate([[0],np.cumsum(deltas>1800)])

        mdf['bout_i']=bout_i
        print 'found',bout_i[-1],'bouts'
        
        print '\n Exporting dataframe as CVS files'
        No_of_rows_per_file=10000
        cvs_dirname='.'.join(_master)
        get_ipython().system('mkdir /mnt/whales_CVS/$cvs_dirname')
        get_ipython().magic('cd /mnt/whales_CVS/$cvs_dirname')
        cvs=0
        _len = mdf.shape[0]
        print 'total length=',_len
        _current_bout=-1
        row_count=0
        file_count=1
        for i in range(_len):
            row=mdf.ix[i,:]
            row_count+=1
            # break the file on bout boundry when length is larger than No_of_rows_per_file
            if (row['bout_i']!=_current_bout):
                _current_bout=row['bout_i']
                if cvs!=0 and (row_count >No_of_rows_per_file):  #close file
                    cvs.close()
                    cvs=0
                    print '\r File: %s: Percent processed %5.2f \tbout_no=%d'%(cvs_filename,(i+0.0)/_len*100,_current_bout),
                    s3helper.local_to_s3(cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename)
                    #!rm $cvs_filename

                if i<_len-1 and cvs==0:  #not last bout. -- open a new file
                    cvs_filename = cvs_dirname+"."+str(file_count)+'.cvs'
                    cvs=open(cvs_filename,'w')
                    row_count=0
                    file_count+=1

            T=mdf.index[i]
            cvs.write(row2cvs(T,row))
        # clean up
        cvs.close()
        cvs=0
        print '\r File: %s: Percent processed %5.2f \tbout_no=%d'%(cvs_filename,(i+0.0)/_len*100,_current_bout),
        s3helper.local_to_s3(cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename)
        #!rm $cvs_filename
        print 'done'
    

print mdf.columns
mdf.head()

get_ipython().system(' wc /mnt/whales_CVS/*/*')

get_ipython().system('head /mnt/whales_CVS/MC.01.Cuviers/MC.01.Cuviers.3.cvs')

shape(TPWS['MSN']),shape(TPWS['MSP'])

akeys

plot(unpackArray(mdf['MSP'][3],data_type=np.float64))

mdf=mdf.sort_index()
mdf[[u'site', u'rec_no', u'species', u'peak2peak']+akeys].head(10)

#find time gaps (breaks) larger the 1800 seconds = 30 minutes
#these time gaps define what is a recording from a single bout of whales.

times=mdf.index

deltas=np.array([(times[i+1]-times[i]).total_seconds() for i in range(len(times)-1)])

bout_i = np.concatenate([[0],np.cumsum(deltas>1800)])

mdf['bout_i']=bout_i

Flags = pd.DataFrame(mdf[akeys])
Flags['ones']=1
Flags.groupby(akeys).count()

plot (bout_i);

plot(times);

mdf.columns

mdf=mdf[[u'site', u'rec_no', u'bout_i', u'species', u'peak2peak','MSN','MSP']+akeys]
mdf.head()

def packed2cvs(row, data_type=np.float64,fmt='%6.4f'):
    return ','.join([fmt%x for x in list(unpackArray(row,data_type=data_type))])
packed2cvs(mdf['MSP'][1])

mdf.columns

row=mdf.ix[0,:]
lrow=list(row)
#print row
pos=1
print "0: time of click"
for i in range(len(row)):
    a=row[i]
    name=row.index[i]
    if type(row[i])!=bytearray:
        print ' %5d: %s\t\t\t\t%s'%(pos,name,str(type(a)))
        pos+=1
    else:
        print " %5d: %s\t\t\t\t an array of length %d"%(pos,name,len(a)/8)
        pos+=len(a)/8

print "total number of fields=",pos

cvs_dirname='.'.join(master_key)
cvs_dirname

cvs=0
_len = mdf.shape[0]
print 'total length=',_len
_current_bout=-1
for i in range(_len):
    row=mdf.ix[i,:]
    if row['bout_i']!=_current_bout:
        if cvs!=0:
            cvs.close()
            s3helper.local_to_s3(cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename)
            get_ipython().system('rm $cvs_filename')

        _current_bout=row['bout_i']
        cvs_filename = cvs_dirname+"."+str(_current_bout)+'.cvs'
        # print cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename
        print '\r Percent processed %5.2f \tbout_no=%d'%((i+0.0)/_len*100,_current_bout),
        cvs=open(cvs_filename,'w')
                                                           
    T=mdf.index[i]
    cvs.write(row2cvs(T,row))
cvs.close()
s3helper.local_to_s3(cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename)
print 'done'

get_ipython().system('pwd')
print cvs_filename
get_ipython().system('ls -l *.cvs')

print cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename
s3helper.local_to_s3(cvs_filename,'CVS/'+cvs_dirname+'/'+cvs_filename)

get_ipython().system('rm DT.01.Cuviers.*')

s3helper.ls_s3('CVS')

get_ipython().system('wc *')

s3helper.local_to_s3('DT.01.Cuviers.cvs101.cvs','CVS/DT.01.Cuviers/DT.01.Cuviers.cvs101.cvs')

for col in ['TPWS1','MD1','FD1','TPWS2','MD2','FD2','TPWS3','MD3','FD3']:
    mdf[col]=False

 

mdf.columns

mdf.head()

mdf.columns

mdf.head()



