#!/usr/bin/python
"""

"""
import sys
sys.path.append('/usr/lib/python2.6/dist-packages')
from mrjob.job import MRJob
import re
from sys import stderr
import pandas as pd
import datetime
import sys

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

def exceptToString(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    return str((exc_type, str(e),fname, exc_tb.tb_lineno))

class MRWeather(MRJob):
        
    def mapper_init(self):
        self.header=["t_mean", "t_count",
                     "dewp", "dewp_count", "slp", "slp_count", "stp","stp_count",
                     "visib", "visib_count", "ws_mean", "ws_count", "ws_max",
                     "ws_gust", "t_max", "t_min", "prcp","snwd", "weather_type"]
        self.keep = ["t_mean","t_max", "t_min", "prcp","snwd"]
        self.bad  = {"t_max":9999.9, "t_min":9999.9, "t_mean":9999.9, "prcp":99.99,"snwd":999.9}
            
    def mapper(self, _, line):
        try:
            F = line.split(',')
            key = "%s-%s"%(F[0],F[1])
            yyyymmdd = F[2]
            dt=datetime.datetime.strptime(F[2],'%Y%m%d')
            year = dt.year
            day_of_year = dt.timetuple().tm_yday 
            F = dict(filter(lambda (k,v): v != nan and v != None,[(k,readDouble(v,self.bad[k])) for k,v in zip(self.header,F[3:]) if (k in self.keep)]))
            C = [F.get(x,0.0) for x in self.keep]
            for key in self.keep:
                if key in F.keys():
                    if not np.isnan(F[key]) and F[key] < 120:
                        yield(key,(1,F[key],F[key]*F[key]))
            if "t_max" in F.keys() and "t_min" in F.keys():
                x = F["t_max"] - F["t_min"]
                if not np.isnan(x):
                    yield("t_var",(1,x,x*x))
        except Exception, e:
            yield (("error","m",exceptToString(e)) , 1)
            
    def combiner(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                N = 0
                C = 0.0
                S = 0.0
                for n,c,s in data:
                    N += n
                    C += c
                    S += s
                yield (key,(N,C,S))
            except Exception, e:
                yield (("error","c",exceptToString(e),), 1)
                
    def reducer(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                N = 0
                C = 0.0
                S = 0.0
                for n,c,s in data:
                    N += n
                    C += c
                    S += s
                yield (key,(N,C,S))
            except Exception, e:
                yield (("error","c",exceptToString(e)), 1)
             
                
if __name__ == '__main__':
    MRWeather.run()