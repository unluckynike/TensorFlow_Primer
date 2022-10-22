'''
@Project ：TensorFlow_Primer 
@File    ：deeplearning.py
@Author  ：hailin
@Date    ：2022/10/22 11:21 
@Info    : 
'''

# import tensorflow as tf

import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():
    """
    tensorflow 基本结构
    :return:
    """
    a = 2
    b = 3
    c = a + b
    print("普通加法运算的结果：\n", c)

    # tensorflow加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("tensorflow加法运算的结果：\n", c_t)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print("c_t_value:\n", c_t_value)
    return None

def graph_demo():
    """
    图演示
    :return:
    """
    a_t=tf.constant(2,name="a_t")
    b_t=tf.constant(3,name="a_t")
    c_t=tf.add(a_t,b_t,name="c_t")
    print("a_t:\n", a_t)
    print("b_t:\n", b_t)
    print("c_t:\n", c_t)

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)

    # 方法2：查看属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)
    # 自定义图

    new_g = tf.Graph()
    # 在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print("a_new:\n", a_new)
        print("b_new:\n", b_new)
        print("c_new:\n", c_new)
        print("a_new的图属性：\n", a_new.graph)
        print("c_new的图属性：\n", c_new.graph)

    # 开启会话
    with tf.Session() as sess:
        # c_t_value = sess.run(c_t)
        # 试图运行自定义图中的数据、操作
        # c_new_value = sess.run((c_new))
        # print("c_new_value:\n", c_new_value)
        print("c_t_value:\n", c_t.eval())
        print("sess的图属性：\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)

    # 开启new_g的会话
    with tf.Session(graph=new_g) as new_sess:
        c_new_value = new_sess.run((c_new))
        print("c_new_value:\n", c_new_value)
        print("new_sess的图属性：\n", new_sess.graph)

    return None

def session_demo():
    """
    会话演示
    :return:
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="a_t")
    c_t = tf.add(a_t, b_t, name="c_t")
    print("a_t:\n", a_t)
    print("b_t:\n", b_t)
    print("c_t:\n", c_t)
    # print("c_t.eval():\n", c_t.eval())

    # 定义占位符
    a_ph = tf.placeholder(tf.float32)
    b_ph = tf.placeholder(tf.float32)
    c_ph = tf.add(a_ph, b_ph)
    print("a_ph:\n", a_ph)
    print("b_ph:\n", b_ph)
    print("c_ph:\n", c_ph)

    # 查看默认图
    # 方法1：调用方法
    default_g = tf.get_default_graph()
    print("default_g:\n", default_g)

    # 方法2：查看属性
    print("a_t的图属性：\n", a_t.graph)
    print("c_t的图属性：\n", c_t.graph)

    # 开启会话
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True)) as sess:
        # 运行placeholder
        c_ph_value = sess.run(c_ph, feed_dict={a_ph: 3.9, b_ph: 4.8})
        print("c_ph_value:\n", c_ph_value)
        # c_t_value = sess.run(c_t)
        # 试图运行自定义图中的数据、操作
        # c_new_value = sess.run((c_new))
        # print("c_new_value:\n", c_new_value)
        # 同时查看a_t, b_t, c_t
        a, b, c = sess.run([a_t, b_t, c_t])
        print("abc:\n", a, b, c)
        print("c_t_value:\n", c_t.eval())
        print("sess的图属性：\n", sess.graph)
        # 1）将图写入本地生成events文件
        tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
    return None

def tensor_demo():
    """
    张量演示
    :return:
    """

    return None

if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()
    # 代码一：tensorflow 基本结构
    # tensorflow_demo()
    # 代码二：图演示
    # graph_demo()
    # 代码三：会话演示
    # session_demo()
    # 代码四：张量演示
    tensor_demo()
