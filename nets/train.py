import numpy as np
import os
import sys
sys.path.append('/usr3/graduate/hongyi/us_crime/us_crime')
import tensorflow as tf
from preprocessing.preprocess import create_dataset, preprocess_for_train
from nets import model
import time



slim = tf.contrib.slim


CITIES = ['chicago', 'lacity', 'stlouis']
DATA_ROOT_DIR = 'LE07_PNG'
CSV_DIR = 'csv_files'

BATCH_SIZE = 16
LR = 0.0001
NUM_EPOCH = 200
DECAY_EPOCH = 180
DECAY_RATE = 2.

BLOCK_1_MODEL = 'block_1_model/block_1.ckpt'
VGG_MODEL = 'vgg_16.ckpt'

SAVE_MODEL_PATH = 'saved_model'
if not os.path.isdir(SAVE_MODEL_PATH):
    os.mkdir(SAVE_MODEL_PATH)

train_log_dir = 'train_log'
if not os.path.isdir(train_log_dir):
    os.mkdir(train_log_dir)

def name_in_checkpoint(var):
  return 'vgg_16/' + var.op.name

def train(log_file_name):
    #create a training dataset
    dataset, iterator, batch_num = create_dataset(DATA_ROOT_DIR, CITIES, CSV_DIR, BATCH_SIZE)
    images, labels = iterator.get_next()

    #preprocess images for training
    img_train = preprocess_for_train(images)
    labels = tf.one_hot(labels,3)

    #run model
    predictions = model.vgg_model(img_train)

    #create loss function
    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    #create optimiazer and train_op
    lr = tf.Variable(LR, trainable=False, name='learning_rate', dtype=tf.float32)
    lr_decay = tf.div(lr, DECAY_RATE)
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = opt.minimize(total_loss)

    #initialization
    vars_to_restore_vgg = slim.get_variables_to_restore(include=['conv2', 'conv3','conv4','conv5','fc6','fc7'])
    vars_to_restore_vgg = {name_in_checkpoint(var):var for var in vars_to_restore_vgg}
    vars_to_restore_fb = slim.get_variables_to_restore(include=['conv1'])

    init_op1, init_feed_dict1 = slim.assign_from_checkpoint(VGG_MODEL, vars_to_restore_vgg)
    init_op2, init_feed_dict2 = slim.assign_from_checkpoint(BLOCK_1_MODEL, vars_to_restore_fb)
    init_op_3 = tf.initializers.variables(slim.get_variables(scope='fc8')+slim.get_variables_by_name('learning_rate'))

    #tf.saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op1, init_feed_dict1)
        sess.run(init_op2, init_feed_dict2)
        sess.run(init_op_3)
        sess.run(iterator.initializer)

        global_step = 0
        for epoch in range(NUM_EPOCH):
            for step in range(batch_num):
                global_step += 1
                if step % global_step != 0:
                    sess.run(train_op)
                else:
                    _, _total_loss = sess.run([train_op,total_loss])
                    f = open(os.path.join(train_log_dir, log_file_name + '.txt'), 'a')
                    f.write('epoch: %d step: %d loss:%.5f\n'%(epoch, global_step, _total_loss))
                    f.close()

            #learning rate decay
            if epoch % 150 == 0 and epoch != 0:
                sess.run(lr_decay)

            #save model
            if epoch % 5 == 0 and epoch != 0:
                saver.save(sess, os.path.join(SAVE_MODEL_PATH,'model.ckpt'),global_step=global_step)



if __name__ == '__main__':
    log_format = time.strftime('%m_%d_%H_%M', time.localtime())
    train(log_format)
































































