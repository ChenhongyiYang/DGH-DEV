import tensorflow as tf
import numpy as np



def vgg_first_block():
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        kernel_rgb = tf.get_variable('kernel_rgb', [3,3,3,64], dtype=tf.float32)

        kernel_15 = tf.get_variable('kernel_15', [3,3,1,64], dtype=tf.float32, initializer=tf.random_normal_initializer())
        kernel_30 = tf.get_variable('kernel_30', [3,3,5,64], dtype=tf.float32, initializer=tf.random_normal_initializer())

        rgb_mean = tf.reduce_mean(kernel_rgb, axis=2)
        value_15 = tf.expand_dims(rgb_mean, axis=2)
        value_30 = tf.concat([value_15]*5, axis=2)

        kernel_15_a = tf.assign(kernel_15, value_15)
        kernel_30_a = tf.assign(kernel_30, value_30)

        biases_1 = tf.get_variable('biases_1', [64], dtype=tf.float32)
        biases_2 = tf.get_variable('biases_2', [64], dtype=tf.float32)
        weights_2 = tf.get_variable('weights_2', [3,3,64,64], dtype=tf.float32)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver({'vgg_16/conv1/conv1_1/weights': kernel_rgb,
                            'vgg_16/conv1/conv1_1/biases': biases_1,
                            'vgg_16/conv1/conv1_2/weights': weights_2,
                            'vgg_16/conv1/conv1_2/biases': biases_2}
                            )
    saver2 = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, '../vgg_16.ckpt')
        sess.run(init)

        sess.run([kernel_15_a, kernel_30_a])

        saver2.save(sess, '../block_1_model/block_1.ckpt')




if __name__ == '__main__':
    vgg_first_block()



























