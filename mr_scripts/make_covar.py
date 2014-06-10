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

#Functions used for reading data/processing
def loads(eVal):
    return pickle.loads(zlib.decompress(base64.b64decode(eVal)))
def dumps(Value):
    return base64.b64encode(zlib.compress(pickle.dumps(Value),9))
def exceptToString(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[0]
    return str((exc_type, fname, exc_tb.tb_lineno))
def infill(tmean,tmax,tmin,prcp,sndp):
    eject = False
    out = []
    for temp in [tmean,tmax,tmin]:
        temp=np.array(temp)
        bad_indexes = np.isnan(temp)
        good_indexes = np.logical_not(bad_indexes)
        good_data = temp[good_indexes]
        interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
        temp[bad_indexes] = interpolated
        out.append(temp.tolist())
    for data in [prcp,sndp]:
        data = np.array(data)
        bad_indexes = np.isnan(data)
        data[bad_indexes] = 0
        out.append(data.tolist())
    return list(itertools.chain.from_iterable(out))

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

class MRWeather(MRJob):
    def mapper(self, _, line):
        try:
            mat = pat.match(line.strip())
            mat.groups()
            station, year, vec = mat.groups()
            year = int(year)
            vec  = loads(vec)
            tmax, tmin, tmean, prcp, sndp = vec[0:365],vec[365:2*365],vec[2*365:3*365],vec[3*365:4*365],vec[4*365:5*365]
            tmax  = [readDouble(x,9999.9) for x in tmax]
            tmin  = [readDouble(x,9999.9) for x in tmin]
            tmean = [readDouble(x,9999.9) for x in tmean]
            prcp  = [readDouble(x,99.99) for x in prcp]
            sndp  = [readDouble(x,999.9) for x in sndp]
            vec = np.matrix(infill(tmean,tmax,tmin,prcp,sndp))
            outer = np.outer(vec,vec)
            yield ("ok",(1,dumps(vec),dumps(outer)))
        except Exception, e:
            yield (("error","m",str(e),exceptToString(e)), 1)
    
    def combiner(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                n, vec, outer = data.next()
                vec = loads(vec)
                outer = loads(outer)
                for np, vecp, outerp in data:
                    n += np
                    vec += loads(vecp)
                    outer += loads(outerp)
                yield ("ok",(n,dumps(vec),dumps(outer)))
            except Exception, e:
                yield (("error","c",exceptToString(e)), 1)
    def reducer(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                n, vec, outer = data.next()
                vec = loads(vec)
                outer = loads(outer)
                for np, vecp, outerp in data:
                    n += np
                    vec += loads(vecp)
                    outer += loads(outerp)
                yield ("ok",(n,dumps(vec),dumps(outer)))
            except Exception, e:
                yield (("error","r",exceptToString(e)), 1)
if __name__ == '__main__':
    MRWeather.run()