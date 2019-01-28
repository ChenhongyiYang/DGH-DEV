from PIL import Image
import numpy as np
import cv2 as cv
import os
from preprocessing.preprocess import read_crime_csv
import matplotlib.pyplot as plt
import glob

channels_30 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7']
channels_15 = ['B8']
cities = ['chicago', 'lacity', 'stlouis']

CSV_DIR = '../csv_files'

def convert_to_png(in_file, out_file):

    img = Image.open(in_file)
    img_np = np.asarray(img)
    img_np = np.multiply(img_np,255).astype(np.int8)
    cv.imwrite(out_file,img_np)


def crime_rate_stat():
    rate_dict = {}
    for fcsv in os.listdir(CSV_DIR):
        extension = os.path.splitext(os.path.join(CSV_DIR, fcsv))[1]
        if extension == '.csv':
            rate_dict.update(read_crime_csv(os.path.join(CSV_DIR, fcsv)))

    print(len(rate_dict))
    labels = ['low','median','high']
    sizes = [0, 0, 0]

    for key in rate_dict:
        var = rate_dict[key]
        if var <= 10:
            sizes[0] += 1
        elif var <= 220:
            sizes[1] += 1
        elif var > 220:
            sizes[2] += 1

    fig, ax = plt.subplots()
    ax.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax.axis('equal')
    plt.show()



def compute_mean(data_root_dir='/Users/yangchenhongyi/Documents/landsat_data/2016/LE07_PNG'):
    mean = {'B1':0., 'B2':0., 'B3':0., 'B4':0., 'B5':0., 'B6_VCID_1':0., 'B6_VCID_2':0., 'B7':0., 'B8':0.}

    num = 0.
    for city in cities:
        dlist_15 = glob.glob(os.path.join(data_root_dir,city,'15')+'/*')
        if '.DS_Store' in dlist_15:
            dlist_15.remove('.DS_Store')
        for dir in dlist_15:
            img = cv.imread(os.path.join(dir,'B8.png'))
            mean['B8'] = (mean['B8'] * num + np.mean(img)) / (num + 1.)
            num += 1.
    print('Scale 15 finished!')
    num = 0.
    for city in cities:
        dlist_30 = glob.glob(os.path.join(data_root_dir,city,'30')+'/*')
        if '.DS_Store' in dlist_30:
            dlist_30.remove('.DS_Store')
        for dir in dlist_30:
            for c in channels_30:
                img = cv.imread(os.path.join(dir,'%s.png'%c))
                mean[c] = (mean[c] * num + np.mean(img)) / (num + 1.)
            num += 1.
    print('Scale 30 finised!')
    print(mean)




def dataset_convert(in_root_path,out_root_path):
    for city in cities:
        path_15 = os.path.join(in_root_path, city, '15')
        path_30 = os.path.join(in_root_path, city, '30')

        out_path_15 = os.path.join(out_root_path, city, '15')
        out_path_30 = os.path.join(out_root_path, city, '30')

        if not os.path.isdir(out_path_15):
            os.makedirs(out_path_15)
        if not os.path.isdir(out_path_30):
            os.makedirs(out_path_30)

        list_15 = os.listdir(path_15)
        list_30 = os.listdir(path_30)

        print(len(list_15))
        print(len(list_30))
        if '.DS_Store' in list_15:
            list_15.remove('.DS_Store')
        if '.DS_Store' in list_30:
            list_30.remove('.DS_Store')

        for geo_id in list_15:
            if not os.path.isdir(os.path.join(out_path_15,geo_id)):
                os.mkdir(os.path.join(out_path_15,geo_id))
            for c in channels_15:
                in_file = os.path.join(path_15, geo_id, '%s.tif'%c)
                out_file = os.path.join(out_path_15, geo_id, '%s.png'%c)
                convert_to_png(in_file, out_file)
        print('Scale 15 has been converted!')

        for geo_id in list_30:
            if not os.path.isdir(os.path.join(out_path_30,geo_id)):
                os.mkdir(os.path.join(out_path_30,geo_id))
            for c in channels_30:
                in_file = os.path.join(path_30, geo_id, '%s.tif' % c)
                out_file = os.path.join(out_path_30, geo_id, '%s.png' % c)
                convert_to_png(in_file, out_file)
        print('Scale 30 has been converted!')


def data_check(data_root_dir='/Users/yangchenhongyi/Documents/landsat_data/2016/LE07_PNG'):
    rate_dict = {}
    for fcsv in os.listdir(CSV_DIR):
        extension = os.path.splitext(os.path.join(CSV_DIR, fcsv))[1]
        if extension == '.csv':
            rate_dict.update(read_crime_csv(os.path.join(CSV_DIR, fcsv)))
    print(len(rate_dict))
    key_list = [key for key in rate_dict]

    key_list_30 = []
    key_list_15 = []
    missing_15 = []
    missing_30 = []
    for city in cities:
        key_list_30 += os.listdir(os.path.join(data_root_dir, city, '30'))
        key_list_15 += os.listdir(os.path.join(data_root_dir, city, '15'))


    for key in key_list:
        if key not in key_list_30:
            missing_30.append(key)
        if key not in key_list_15:
            missing_15.append(key)

    '''
    for key in key_list_30:
        if key not in key_list:
            missing_30.append(key)
    for key in key_list_15:
        if key not in key_list:
            missing_15.append(key)
    '''


    print(len(key_list))
    print(len(key_list_30))
    print(len(key_list_15))

    while '.DS_Store' in missing_30:
        missing_30.remove('.DS_Store')
    print(missing_30)
    print(missing_15)








if __name__ == '__main__':
    '''
    in_root_path = '/Users/yangchenhongyi/Documents/landsat_data/2016/LE07'
    out_root_path = '/Users/yangchenhongyi/Documents/landsat_data/2016/LE07_PNG'
    dataset_convert(in_root_path,out_root_path)
    '''
    #crime_rate_stat()
    #compute_mean(   )
    data_check()



























