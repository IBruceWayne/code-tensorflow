# -*- coding: utf-8 -*-
# @author: gcg
import tensorflow as tf

sess = tf.InteractiveSession()

a = tf.Variable([[1, 2, 6, 0], [3, 1, 5, 7], [2, 6, 9, 1]])
b = tf.Variable(tf.random_uniform([2, 3, 4], 1, 5, dtype=tf.int32))
c = tf.argmax(a, 0)
d = tf.argmax(a, 1)
e = tf.argmax(b, 0)
f = tf.argmax(b, 1)


sess.run(tf.global_variables_initializer())
print('a:\n', sess.run(a))
print('b:\n', sess.run(b))
print('\ntf.argmax(a, 0):\n', sess.run(c))
print('\ntf.argmax(a, 1):\n', sess.run(d))
print('\ntf.argmax(b, 0):\n', sess.run(e))
print('\ntf.argmax(b, 1):\n', sess.run(f))


'''总结
tf.argmax 返回最大元素的下标。
tf.arg_max功能相同，但不赞成使用。因为处理相同的矩阵，多次操作返回结果不保证一致性。
tf.argmax(a, 0) 行与行之间对比。
    [1 2 6 0]
    [3 1 5 7]
    [2 6 9 1]
结果：1 2 2 1
tf.argmax(a, 1) 列与列之间对比。
               结果
    [1 2 6 0]   2
    [3 1 5 7]   3
    [2 6 9 1]   2
'''