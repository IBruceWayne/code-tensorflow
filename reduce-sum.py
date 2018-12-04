# -*- coding: utf-8 -*-
# @author: gcg
# tf.reduce_sum()
import tensorflow as tf
sess = tf.InteractiveSession()

# 应用在二维数组上
a = tf.Variable(tf.random_uniform([2, 3], 1, 4, dtype=tf.int32))
total_sum = tf.reduce_sum(a)  # 默认返回所有元素的和
indices0 = tf.reduce_sum(a, [0])
indices01 = tf.reduce_sum(a, [0, 1])
indices11 = tf.reduce_sum(a, [1, 1])
# 应用在三维数组上
b = tf.Variable(tf.random_uniform([2, 3, 4], 1, 5, dtype=tf.int32))
total_sum_b = tf.reduce_sum(b)  # 默认返回所有元素的和
indices0_b = tf.reduce_sum(b, [0])
indices01_b = tf.reduce_sum(b, [0, 1])
# tf.reduce_mean()
ma = tf.Variable(tf.random_uniform([2, 3], 1, 4, dtype=tf.float32))
mean_total_sum = tf.reduce_mean(ma)  # 默认返回所有元素的和
mean_indices0 = tf.reduce_mean(ma, [0])
mean_indices01 = tf.reduce_mean(ma, [0, 1])

sess.run(tf.global_variables_initializer())
# 二维数组操作结果
print('二维数组：\n', sess.run(a))
print('未指定任何缩减维度，默认返回所有元素的和，tf.reduce_sum(a)：', sess.run(total_sum))
print('指定缩减第一个维度，则第二个维度默认保留，tf.reduce_sum(a,[0])：', sess.run(indices0))
print('指定缩减第一、第二个维度，tf.reduce_sum(a,[0, 1])：', sess.run(indices01))
print('指定缩减第二个维度，tf.reduce_sum(a,[1, 1])：', sess.run(indices11))

# 三维数组操作结果
print('\n三维数组, 形状 [2, 3, 4]：\n', sess.run(b))
print('未指定任何缩减维度，默认返回所有元素的和，tf.reduce_sum(a)：', sess.run(total_sum_b))
print('指定缩减第一个维度，则剩余维度默认保留，tf.reduce_sum(a,[0])：\n', sess.run(indices0_b))
print('指定缩减第一个维度，第二个维度，则剩余维度默认保留，tf.reduce_sum(a,[0, 1])：\n', sess.run(indices01_b))

# 平均值
print('\n数组：\n', sess.run(ma))
print('未指定任何缩减维度，默认返回所有元素的平均值，tf.reduce_mean(a)：', sess.run(mean_total_sum))
print('指定缩减第一个维度，则第二个维度默认保留，tf.reduce_mean(a,[0])：', sess.run(mean_indices0))
print('指定缩减第一、第二个维度，tf.reduce_mean(a,[0, 1])：', sess.run(mean_indices01))

'''总结
tf.reduce_sum 按维度缩减，0代表缩减第一个维度，而缩减其他维度则用1表示，默认返回一个标量，代表所有元素的和
例如：tf.reduce_sum(a, [0, 1]) 第一0表示缩减第一个维度， 第二个1表示缩减第二个维度，如果是三维以上的数组，其余维度默认保留。
为什么用0表示缩减第一个维度？这是因为python中axis参数默认用0表示按列求和，用1表示按行求和。tensorflow也支持用axis索引。
tf.reduce_sum(a, [0]) == tf.reduce_sum(a, 0)/tf.reduce_sum(a, axis=0)
tf.reduce_sum(a, [1]) == tf.reduce_sum(a, 1)
keep_dims 意味维度保持不变。维度指的是多维数组中，索引一个元素至少需要的下标数量。keep_dims=True意味着在缩减的维度上套一个中括号。
值得一提的是,[1,2,3,4]是一维的。
其他按维度缩减的函数，功能类似，如：tf.reduce_mean
更多信息：https://www.zhihu.com/question/51325408
'''