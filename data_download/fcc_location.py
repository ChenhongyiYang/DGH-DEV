#!/usr/bin/env python

# fcc-lat-lon-county
# copyright (C) 2018 J.W. Crockett, Jr.
# basic lib to query the FCC Census Area API with a latitude and longitude and get
# back what US state and county or county-level division it belongs to, with simple
# rate limiting to help avoid the banhammer.
# Issue: right now assumes that if it sent two numbers in and no results came back,
# it's a non-US location. Actual error handling TK.
# see https://geo.fcc.gov/api/census/#!/area/get_area for API docs.

import urllib2, json, numbers, time

HOST_URL = "https://geo.fcc.gov/"

RATE_PER_SEC = 1
last_call_time = None


def location(lat, lon):
    global last_call_time

    if not (isinstance(lat, numbers.Real) and isinstance(lon, numbers.Real)):
        raise BadLatLonException

    ENDPOINT = "api/census/area"

    if last_call_time:
        delay = time.time() - last_call_time
        if (delay < (1.0 / RATE_PER_SEC)):
            time.sleep(delay)
        last_call_time = time.time()

    last_call_time = time.time()

    url = HOST_URL + ENDPOINT + "?lat=" + str(lat) + "&lon=" + str(lon) +'&format=json'
    #print(url)

    fh = urllib2.urlopen(HOST_URL + ENDPOINT + "?lat=" + str(lat) + "&lon=" + str(lon) +'&format=json')
    jjson = json.loads(fh.read())
    bbox = jjson['results'][0]['bbox']
    return bbox

def read_txt(txt_file):
    #read file
    f = open(txt_file,'r')
    lines = f.readlines()
    f.close()

    all_data = []
    for i in range(1,len(lines)):
        line = lines[i].split('\t')
        if len(line) == 0:
            return
        data = {}
        data['id'] = line[1]
        data['geo_cent'] = [float(line[-1]),float(line[-2])]
        all_data.append(data)

    return all_data

def read_and_write(all_data,filename):
    i = 1
    for data in all_data:
        print(i,data['id'])
        lat, lon = data['geo_cent'][1], data['geo_cent'][0]
        bbox = location(lat,lon)
        data['bbox'] = bbox
        i += 1

    f = open(filename,'w')
    for data in all_data:
        f.write(data['id'] + ' ' + str(data['bbox'][0]) + ' ' + str(data['bbox'][1]) + ' ' + str(data['bbox'][2])+ ' ' + str(data['bbox'][3])+'\n')
    f.close()





if __name__ == '__main__':
    txt_file = 'csv_files/lacity_area.txt'
    out_file = 'csv_files/lacity_bbox.txt'
    all_data = read_txt(txt_file)
    read_and_write(all_data,out_file)

