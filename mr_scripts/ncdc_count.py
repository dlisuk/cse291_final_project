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

class MRWeather(MRJob):
    def configure_options(self):
        super(MRWeather,self).configure_options()
        self.add_file_option('--stations')
        
    def init(self):
        self.keep = ["t_mean","dewp","slp","stp","visib","ws_mean","ws_max",
                     "ws_gust", "t_max", "t_min", "prcp","snwd", "weather_type"]
    def mapper_init(self):
        self.header=["t_mean", "t_count",
                     "dewp", "dewp_count", "slp", "slp_count", "stp","stp_count",
                     "visib", "visib_count", "ws_mean", "ws_count", "ws_max",
                     "ws_gust", "t_max", "t_min", "prcp","snwd", "weather_type"]
        self.keep = ["t_mean","dewp","slp","stp","visib","ws_mean","ws_max",
                     "ws_gust", "t_max", "t_min", "prcp","snwd", "weather_type"]
        self.bad_pat = re.compile("[9\.]*$")
        df = pd.read_csv(self.options.stations)
        self.stations = {}
        for _,row in df.iterrows():
            key = "%06d-%05d"%(row['USAF'],row['WBAN'])
            val = (float(row["LAT"])/1000,float(row["LON"])/1000,float(row["ELEV(.1M)"]))
            self.stations[key] = val
    def combiner_init(self):
        self.init()
    def reducer_init(self):
        self.init()
            
    def mapper(self, _, line):
        try:
            F = line.split(',')
            key = "%s-%s"%(F[0],F[1])
            yyyymmdd = F[2]
            dt=datetime.datetime.strptime(F[2],'%Y%m%d')
            year = dt.year
            day_of_year = dt.timetuple().tm_yday 
            F = dict(filter(lambda (k,v): ("count" not in k) and (not self.bad_pat.match(v)), zip(self.header,F[2:])))
            C = [int(x in F.keys()) for x in self.keep]
            (lat,lon,elevation) = self.stations[key]
            yield ((lat,lon),C)            
        except Exception, e:
            yield (("error","m",str(e)), 1)
            
    def combiner(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                C = [0 for x in self.keep]
                for rec in data:
                    for i in range(len(rec)):
                        C[i] += rec[i]
                yield (key,C)
            except Exception, e:
                yield (("error","c",str(e)), 1)
    def reducer(self, key, data):
        if key[0] == "error":
            yield (key,sum(data))
        else:
            try:
                C = [0 for x in self.keep]
                for rec in data:
                    for i in range(len(rec)):
                        C[i] += rec[i]
                yield (key,C)
            except Exception, e:
                yield (("error","r",str(e)), 1)
if __name__ == '__main__':
    MRWeather.run()