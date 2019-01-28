import csv
import ee
import datetime
import urllib.request
import os
import glob
import zipfile
import time
import matplotlib.pyplot as plt
import numpy as np
from preprocessing.preprocess import read_crime_csv



geo_parameters = {
    'radius':{'chicago':6368593., 'lacity':6371080., 'stlouis':6369278.},
    'latitude': {'chicago':0.715584, 'lacity':0.610852, 'stlouis':0.715585}
}

min_leng = 900.


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

def read_bbox(all_data,bbox_file):
    f = open(bbox_file,'r')
    lines = f.readlines()
    f.close()

    records = {}
    for line in lines:
        line = line.strip().split(' ')
        if len(line) != 5:
            continue
        key = line[0]
        val = [float(line[1]),float(line[2]),float(line[3]),float(line[4])]
        records[key] = val

    for data in all_data:
        id = data['id']
        data['bbox'] = records[id]
    return all_data

def read_bbox2(crime_dict,bbox_file):
    f = open(bbox_file, 'r')
    lines = f.readlines()
    f.close()

    records = {}
    for line in lines:
        line = line.strip().split(' ')
        if len(line) != 5:
            continue
        key = line[0]
        val = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        records[key] = val

    ret_dict = {}
    for key in crime_dict:
        ret_dict[key] = records[key]
    return ret_dict



def download_geo1(geo_location,id,city,root_path,scale,dataset):
    save_path = root_path + '/' + city + '/' + str(scale)
    save_path_zip = root_path + '/' + city + '/' + 'zip'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_zip):
        os.makedirs(save_path_zip)

    if scale == 30:
        channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7', 'B8']
    elif scale == 15:
        channels = ['B8']
    else:
        channels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7', 'B8']

    collection = (ee.ImageCollection(dataset).filterDate(datetime.datetime(2016, 1, 1), datetime.datetime(2016, 12, 31)))
    image = collection.median()

    #expand the block
    r = geo_parameters['radius'][city]
    theta = geo_parameters['latitude'][city]
    latibound = (min_leng * 180.) / (np.pi * r)
    longbound = (min_leng * 180.) / (np.pi * r * np.cos(theta))
    xmin, ymin, xmax, ymax = geo_location[0], geo_location[1], geo_location[2], geo_location[3]
    if xmax-xmin < longbound:
        xmean = (xmax + xmin) / 2.
        xmin = xmean - longbound / 2.
        xmax = xmean + longbound / 2.

    if ymax-ymin < latibound:
        ymean = (ymax + ymin) / 2.
        ymin = ymean - latibound / 2.
        ymax = ymean + latibound / 2.
    new_box = [xmin, ymin, xmax, ymax]


    clipped_image = image.clip(ee.Geometry.Rectangle(new_box))

    url = clipped_image.getDownloadURL({'scale': scale, 'crs': 'EPSG:4326'})
    spatial_data_file = save_path_zip + '/' + id + '_%d.zip'%scale
    a = urllib.request.urlretrieve(url, spatial_data_file)
    zfile = zipfile.ZipFile(spatial_data_file)
    zfile.extractall(save_path + '/' + id)

    for filename in os.listdir(save_path+'/'+id):

        band,extension = filename.split('.')[-2], filename.split('.')[-1]
        if band in channels:
            os.rename(
                save_path + '/' + id + '/' + filename,
                 save_path + '/' + id + '/' + band + '.' + extension)
        else:
            os.remove(save_path + '/' + id + '/' + filename)
    os.remove(spatial_data_file)



def draw_scatter(all_data):
    Xs = []
    Ys = []
    for data in all_data:
        Xs.append(data['geo_cent'][0])
        Ys.append(data['geo_cent'][1])
    plt.scatter(Xs,Ys,)
    plt.show()



if __name__ == '__main__':
    ee.Initialize()

    #initialize all config parameters

    root_path = '/Users/yangchenhongyi/Documents/landsat_data/2016'
    dataset = 'LANDSAT/LE07/C01/T1_TOA'
    root_path = root_path + '/' + dataset.split('/')[1]
    city = 'lacity'

    if not os.path.isdir(root_path + '/' + city):
        os.mkdir(root_path + '/' + city)

    csv_file = '../csv_files/' + city + '_tract_crime.csv'
    txt_file = '../txt_files/' + city + '_area.txt'
    bbox_file = '../txt_files/' + city + '_bbox.txt'

    #all_data = read_txt(txt_file)
    #all_data = read_bbox(all_data,bbox_file)
    #print(all_data[0])
    crime_dict = read_crime_csv(csv_file)
    bbox_dict = read_bbox2(crime_dict, bbox_file)

    exist_list_15 = glob.glob(root_path + '/' + city + '/' + '15' + '/*')
    exist_15 = [var.split('/')[-1] for var in exist_list_15]
    print(exist_list_15)

    num = 0
    for key in bbox_dict:
        num += 1
        if key not in exist_15:
            print(num, key)
            try:
                download_geo1(bbox_dict[key], key, city, root_path, 30, dataset)
            except:
                time.sleep(300)
            try:
                download_geo1(bbox_dict[key], key, city, root_path, 15, dataset)
            except:
                time.sleep(300)
            '''
            try:
                download_geo1(data['bbox'], data['id'], city, root_path, 60, dataset)
            except:
                time.sleep(300)
            '''
































