# -*- coding: utf-8 -*-
# @author: gcg
import time
import tensorflow as tf



# # 建立一个读入数据的generator，然后使用tf.data.Dataset对其进行包装转换
# def gen():
#   f = open('dta.txt', 'r', encoding='utf-8')
#   with open('data.txt', encoding='utf-8') as f:
#     lines = [line.strip().split() for line in f.readlines()]
#   index = 0
#   while True:
#     f = open('data.txt', 'r', encoding='utf-8')
#     sentence = lines[index][1::]
#     label = lines[index][0]
#     yield (sentence, label)
#     index += 1
#     if index == len(lines):
#       index = 0
#
#
# def create_dataset():
#
#   data = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
#                                         (tf.TensorShape([10]), tf.TensorShape([])))
#   data = data.shuffle(buffer_size=3)
#   data = data.batch(3)
#   #data.padded_batch(batch_size=3, padded_shapes=([10], [None]))
#   data = data.make_one_shot_iterator()
#   tt = time.time()
#   with tf.Session() as sess:
#     for i in range(10):
#       sent, labels = data.get_next()
#       sent, labels = sess.run([sent, labels])
#       print('{} -> {} -> {}'.format(i, labels, sent))
#       print(time.time() - tt)
#
#
# def main():
#   create_dataset()
#
# # 首先构建一个generator：gen，然后使用tf.data.Dataset的from_generator函数，通过指定数据类型，数据的shape等参数，
# # 构建一个Dataset，当然，随后也要指定一下batch_size，最后使用make_one_shot_iterator()函数，构建一个iterator
#
# if __name__ == '__main__':
#   main()