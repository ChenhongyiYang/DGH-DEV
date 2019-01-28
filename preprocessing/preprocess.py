import tensorflow as tf
import numpy as np
import csv
import os

slim = tf.contrib.slim

dtype = tf.float32

INPUT_SIZE = [224,224]
SCALE = [15, 30]

channels_30 = ['B3', 'B2', 'B1', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7']
channels_15 = ['B8']

#{'B3':37.5, 'B2':36.8, 'B1':39.2, 'B4':54.3, 'B5':48.1, 'B6_VCID_1':32.2, 'B6_VCID_2':32.7, 'B7':35.2, 'B8':45.5}
_MEAN_RGB = [37.5, 36.8, 39.2]
_MEAN_30 = [54.3, 48.1, 32.2, 32.7, 35.2]
_MEAN_15 = [45.5]

#============================================================================
#                            deal with datas
#============================================================================

def read_crime_csv(filename):
    crime_dict = {}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            geo_id = row[0]
            crime = int(row[1])
            crime_dict[geo_id] = crime
            line_count += 1
    print('%d lines have been read!'%line_count)
    return crime_dict

def get_data(root_path,city,crime_dict):
    def convert_label(rate,b1=10,b2=220):
        if rate <= b1:
            label = 0
        elif rate <= b2:
            label = 1
        else:
            label = 3
        return label

    data_path_15 = os.path.join(root_path, city,'15')
    data_path_30 = os.path.join(root_path, city, '30')

    labels = []
    flist = []
    id_list = os.listdir(data_path_15)
    while '.DS_Store' in id_list:
        id_list.remove('.DS_Store')

    for geo_id in id_list:
        fimgs_30 = [data_path_30+'/'+geo_id+'/'+c+'.png' for c in channels_30]
        fimgs_15 = [data_path_15+'/'+geo_id+'/'+c+'.png' for c in channels_15]
        flist.append(fimgs_30+fimgs_15)
        labels.append(convert_label(crime_dict[geo_id]))

    return flist, labels



#============================================================================
#                    tensorflow dataset and preprocessing
#============================================================================

def _parse_func(filenames):
    img_list_30 = []
    for i in range(8):
        image_string = tf.read_file(filenames[i])
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize_images(image_decoded, tf.shape(image_decoded)[:2] * 2, method=1)
        image_resized = tf.image.resize_images(image_resized, INPUT_SIZE)
        img_list_30.append(image_resized)
    tensor_RGB = tf.concat(img_list_30[:3], axis=-1)
    tensor_30 = tf.concat(img_list_30[3:], axis=-1)

    image_string = tf.read_file(filenames[8])
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, INPUT_SIZE)
    tensor_15 = image_resized
    all_bands = tf.concat([tensor_RGB,tensor_30,tensor_15],axis=-1)

    return all_bands


def dataset_gener(flist, labels, batch_size):
    dx = tf.data.Dataset.from_tensor_slices(flist)
    dy = tf.data.Dataset.from_tensor_slices(labels)

    dx = dx.map(_parse_func)
    data = tf.data.Dataset.zip((dx,dy)).shuffle(500).batch(batch_size=batch_size).repeat()
    iterator = data.make_initializable_iterator()
    return data, iterator


def create_dataset(data_root_dir, cities, csv_dir, batch_size=1):
    csv_list = os.listdir(csv_dir)
    crime_dict = {}
    for fcsv in csv_list:
        extension = os.path.splitext(os.path.join(csv_dir,fcsv))[1]
        if extension =='.csv':
            crime_dict.update(read_crime_csv(os.path.join(csv_dir,fcsv)))

    flist, labels = [], []
    for city in cities:
        city_flist, city_labels = get_data(data_root_dir, city, crime_dict)
        flist = flist + city_flist
        labels = labels + city_labels

    data, iterator = dataset_gener(flist, labels, batch_size)
    batch_num = len(flist) // batch_size
    return data, iterator, batch_num



def preprocess_for_train(img):
    bands_RGB, bands_30, bands_15 = tf.split(img, [3,5,1], axis=-1)
    #whiten images
    mean_RGB = tf.constant(_MEAN_RGB, dtype=dtype)
    mean_30 = tf.constant(_MEAN_30, dtype=dtype)
    mean_15 = tf.constant(_MEAN_15, dtype=dtype)

    bands_RGB = bands_RGB - mean_RGB
    bands_30 = bands_30 - mean_30
    bands_15 = bands_15 - mean_15

    #random flip, up to down & left to right
    image = tf.concat([bands_RGB, bands_30, bands_15], axis=-1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    #bands_RGB, bands_30, bands_15 = tf.split(image, [3, 5, 1], axis=-1)
    #all_bands = tf.concat([bands_RGB, bands_30, bands_15], axis=-1)
    return image






if __name__ == '__main__':
    '''
    flist_30 = ['/Users/yangchenhongyi/Documents/landsat_data/LE07/stlouis/30/29001950200/'+c+'.png' for c in channels_30]
    flist_15 = ['/Users/yangchenhongyi/Documents/landsat_data/LE07/stlouis/15/29001950200/'+c+'.png' for c in channels_15]
    flist = flist_30 + flist_15

    image_files = [flist]
    dx = tf.data.Dataset.from_tensor_slices(image_files).repeat()
    dx = dx.map(map_func=_parse_func)
    iterator = dx.make_initializable_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(2):
            val = sess.run(next_element)
            print('val0')
            print(val[0].shape)
            print('val1')
            print(val[1].shape)
            print('val2')
            print(val[2].shape)
    '''
    crime_dict = read_crime_csv('../csv_files/stlouis_tract_crime.csv')
    print(crime_dict)







































