# -*- coding: utf-8 -*-
# @author: gcg
# tf.nn.softmax(logits, dim, name) dim默认等于1，注意这里的log底数是e，约等于2.71828
# tf.nn.softmax_cross_entropy_with_logits(logits,labels,name=None) labels是one-hot向量计算交叉熵
# 上一个函数逐渐被启用，推荐使用tf.nn.softmax_cross_entropy_with_logits_v2()
# tf.nn.sparse_softmax_cross_entropy_with_logits()
# 上述函数接收的logits，每一个都只能属于一类(互斥的)，但是labels可以是一个概率分布如：[0.1, 0.9]
# 但它必须是一个有效的概率分布，和必须等于1，否则损失会计算错误。
import tensorflow as tf
sess = tf.InteractiveSession()

a = tf.Variable(tf.random_uniform([2, 3], 0, 1))
asoftmax = tf.nn.softmax(a)  # 接受未归一化的数据进行归一化，变成概率。logits意味着未归一化的数据。形状为[batch, nclass]
sum_asoftmax = tf.reduce_sum(asoftmax, 1)
#
b = tf.Variable(tf.random_uniform([2, 3], 0, 1))  # batch=2，nclass=3
b_labels = tf.Variable(tf.random_uniform([2], 0, 3, dtype=tf.int32))  # 类别号从0开始：0-1-2 三类
b_labels2 = tf.Variable(tf.constant([[0.2, 0.1, 0.7], [0.5, 0.2, 0.3]]))  # 类别号从0开始 标签不需要是one-hot也可以是一个有效的概率分布
b_labels_one_hot = tf.one_hot(b_labels, depth=3)
b_cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=b, labels=b_labels2)
b_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=b, labels=b_labels_one_hot)
# bsoftmaxv2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=b, labels=b_labels_one_hot
b_softmax = tf.log(tf.nn.softmax(b))
t = -tf.reduce_sum(b_labels2 * b_softmax, 1)  # 这个应该等于b_cross_entropy2
# 不需要把标签转换为one-hot向量
sparse_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=b, labels=b_labels)
# 结果
sess.run(tf.global_variables_initializer())
# tf.nn.softmax
print('tf.nn.softmax：')
print('a:\n', sess.run(a))
print('softmax(a):\n', sess.run(asoftmax))
print('sum-softmax(a):\n', sess.run(sum_asoftmax))
# tf.nn.softmax_cross_entropy_with_logits()
print('\ntf.nn.softmax_cross_entropy_with_logits():')
print('b:\n', sess.run(b))
print('b-labels:\n', sess.run(b_labels))
print('b-labels-one-hot:\n', sess.run(b_labels_one_hot))
print('ln(b_softmax):\n', sess.run(b_softmax))
print('bsoftmax-cross-entropy:\n', sess.run(b_cross_entropy))
print('bsoftmax-cross-entropy2:\n', sess.run(b_cross_entropy2))
print('手工计算交叉熵：\n', sess.run(t))
#
print('sparse-cross-entropy:\n', sess.run(sparse_cross_entropy))