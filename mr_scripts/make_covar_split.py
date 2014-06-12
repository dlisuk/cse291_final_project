#!/usr/bin/python
import sys
sys.path.append('/usr/lib/python2.6/dist-packages')
from mrjob.job import MRJob
#Standard imports
import pickle
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from pandas.tools import rplot
import re
import zlib
import base64 
import traceback, os.path
import itertools
import cPickle

#Functions used for reading data/processing
def loads(eVal):
    return pickle.loads(zlib.decompress(base64.b64decode(eVal)))
def dumps(Value):
    return base64.b64encode(zlib.compress(pickle.dumps(Value),9))
def exceptToString(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[0]
    return str((str(e), exc_type, fname, exc_tb.tb_lineno))
def infill(tmean,tvar,prcp,sndp):
    eject = False
    out = []
    for temp in [tmean,tvar]:
        temp=np.array(temp)
        bad_indexes = np.isnan(temp)
        good_indexes = np.logical_not(bad_indexes)
        good_data = temp[good_indexes]
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        temp[bad_indexes] = interpolated
        out.append(smooth(5,temp.tolist()))
    for data in [prcp,sndp]:
        data = np.array(data)
        bad_indexes = np.isnan(data)
        data[bad_indexes] = 0
        out.append(smooth(10,data.tolist()))
    return out
def smooth(t,xs):
    out = []
    for i in range(0,len(xs),t):
        x = sum(xs[i:(i+t)])/len(xs[i:(i+t)])
        out.append(x)
    return out

nan = float("NaN")
def readDouble(x,z):
    try:
        y = float(x)
    except ValueError:
        try:
            ##In this case, we have a non, probalby there is a flag here that needs to be removed
            y = float(x.translate(None,'ABCDEFGHI*'))
        except ValueError:
            ##We want to print any bad values that get to here so we can analyze further
            print(x)
            return nan
    #Missing data goes to nan
    if abs(y-z) < 0.01:
        return nan
    return y

pat = re.compile('"(\d*):(\d\d\d\d)"\t"(.*)"$')

mean_std = {'prcp': (0.0677889216399339, 0.25516420874100204),
 't_var': (16.27944732297074, 8.9305688654860766),
 'snwd': (12.013581395348849, 10.383514291128623),
 'temp': (53.25642013353873, 22.662544254342716)}
def z_scale(key,val):
    mu,sig = mean_std[key]
    return (val - mu)/sig
class MRWeather(MRJob):
    def configure_options(self):
        super(MRWeather,self).configure_options()
        self.add_file_option('--splits')
 
    def mapper_init(self):
        f = open( self.options.splits, "rb" )
        pickleFile = cPickle.Unpickler( f )
        self.ncdc_partition = pickleFile.load()
        f.close()

    def mapper(self, _, line):
        
        try:
            mat = pat.match(line.strip())
            mat.groups()
            station, year, vec = mat.groups()
            station = int(station)
            year = int(year)
            vec  = loads(vec)
            key = self.ncdc_partition[station]
            #We want to discard a day from leap years
            if(len(vec)==1830):
                tmax, tmin, tmean, prcp, sndp = vec[0:366],vec[366:2*366],vec[2*366:3*366],vec[3*366:4*366],vec[4*366:5*366]
                tmax, tmin, tmean, prcp, sndp = tmax[0:365], tmin[0:365], tmean[0:365], prcp[0:365], sndp[0:365]
            elif(len(vec)==1835):
                tmax, tmin, tmean, prcp, sndp = vec[0:367],vec[367:2*367],vec[2*367:3*367],vec[3*367:4*367],vec[4*367:5*367]
                tmax, tmin, tmean, prcp, sndp = tmax[0:365], tmin[0:365], tmean[0:365], prcp[0:365], sndp[0:365]
            elif(len(vec)==1825):
                tmax, tmin, tmean, prcp, sndp = vec[0:365],vec[365:2*365],vec[2*365:3*365],vec[3*365:4*365],vec[4*365:5*365]
            else:
                raise ValueError("len(vec)="+str(len(vec)))
            tvar  = [z_scale("t_var",readDouble(x,9999.9)-readDouble(y,9999.9)) for (x,y) in zip(tmax,tmin)]
            tmean = [z_scale("temp",readDouble(x,9999.9)) for x in tmean]
            prcp  = [z_scale("prcp",readDouble(x,99.99)) for x in prcp]
            sndp  = [z_scale("snwd",readDouble(x,999.9)) for x in sndp]
            xs    = infill(tmean,tvar,prcp,sndp)
            tmean = np.matrix(xs[0])
            tvar  = np.matrix(xs[1])
            prcp  = np.matrix(xs[2])
            sndp  = np.matrix(xs[3])
            yield (("ok",key),(1,dumps(tmean),None,dumps(tvar),None,dumps(prcp),None,dumps(sndp),None))
        except Exception, e:
            yield (("error","m",str(e),exceptToString(e)), 1)
    
    def combiner(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                n, x1, xo1, x2, xo2, x3, xo3, x4, xo4 = data.next()
                x1 = loads(x1)
                x2 = loads(x2)
                x3 = loads(x3)
                x4 = loads(x4)
                if xo1 == None:
                    xo1 = np.outer(x1,x1)
                else:
                    xo1 = loads(xo1)
                    
                if xo2 == None:
                    xo2 = np.outer(x2,x2)
                else:
                    xo2 = loads(xo2)
                    
                if xo3 == None:
                    xo3 = np.outer(x3,x3)
                else:
                    xo3 = loads(xo3)
                    
                if xo4 == None:
                    xo4 = np.outer(x4,x4)
                else:
                    xo4 = loads(xo4)
                    
                for enc in data:
                    nP, x1P, xo1P, x2P, xo2P, x3P, xo3P, x4P, xo4P = enc
                    x1P = loads(x1P)
                    x2P = loads(x2P)
                    x3P = loads(x3P)
                    x4P = loads(x4P)
                    if xo1P == None:
                        xo1P = np.outer(x1P,x1P)
                    else:
                        xo1P = loads(xo1P)

                    if xo2P == None:
                        xo2P = np.outer(x2P,x2P)
                    else:
                        xo2P = loads(xo2P)

                    if xo3P == None:
                        xo3P = np.outer(x3P,x3P)
                    else:
                        xo3P = loads(xo3P)

                    if xo4P == None:
                        xo4P = np.outer(x4P,x4P)
                    else:
                        xo4P = loads(xo4P)

                    n += nP
                    x1 += x1P
                    xo1 += xo1P
                    x2 += x2P
                    xo2 += xo2P
                    x3 += x3P
                    xo3 += xo3P
                    x4 += x4P
                    xo4 += xo4P
                yield (key,dumps((n, x1, xo1, x2, xo2, x3, xo3, x4, xo4)))
            except Exception, e:
                yield (("error","c",exceptToString(e)), 1)
    def reducer(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                n, x1, xo1, x2, xo2, x3, xo3, x4, xo4 = loads(data.next())
                for enc in data:
                    nP, x1P, xo1P, x2P, xo2P, x3P, xo3P, x4P, xo4P = loads(enc)
                    n += nP
                    x1 += x1P
                    xo1 += xo1P
                    x2 += x2P
                    xo2 += xo2P
                    x3 += x3P
                    xo3 += xo3P
                    x4 += x4P
                    xo4 += xo4P
                yield (key,dumps((n, x1, xo1, x2, xo2, x3, xo3, x4, xo4)))
            except Exception, e:
                yield (("error","r",exceptToString(e)), 1)
if __name__ == '__main__':
    MRWeather.run()