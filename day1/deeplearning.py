'''
@Project ：TensorFlow_Primer 
@File    ：deeplearning.py
@Author  ：hailin
@Date    ：2022/10/22 11:21 
@Info    : 图 会话 张量 变量
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
    a_t = tf.constant(2, name="a_t")
    b_t = tf.constant(3, name="a_t")
    c_t = tf.add(a_t, b_t, name="c_t")
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
    tensor1 = tf.constant(4.0)
    tensor2 = tf.constant([1, 2, 3, 4])
    linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)
    print("tensor1:\n", tensor1)  # 标量
    print("tensor2:\n", tensor2)  # 向量
    print("linear_squares:\n", linear_squares)  # 矩阵 （4，1）4行1列

    print("zeros", tf.zeros(5))
    # 张量的类型修改
    l_cast = tf.cast(linear_squares, dtype=tf.float32)
    print("l_cast:\n", l_cast)

    # 张量形状改变
    # 更新/改变静态形状
    # 定义占位符
    # 没有完全固定下来的静态形状
    a_p = tf.placeholder(dtype=tf.float32, shape=[None, None])
    b_p = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    c_p = tf.placeholder(dtype=tf.float32, shape=[3, 2])
    print("a_p:\n", a_p)
    print("b_p:\n", b_p)
    print("c_p:\n", c_p)
    # 更新形状未确定的部分
    # a_p.set_shape([2, 3])
    # b_p.set_shape([2, 10])
    # c_p.set_shape([2, 3]) # 报错

    # 查看更新后的形状
    print("a_p:\n", a_p)
    print("b_p:\n", b_p)
    print("c_p:\n", c_p)

    # 动态修改形状
    print("-------动态修改形状------")
    a_p_reshape = tf.reshape(a_p, shape=[2, 3, 1])
    print("a_p:\n", a_p)
    # print("b_p:\n", b_p)
    print("a_p_reshape:\n", a_p_reshape)
    c_p_reshape = tf.reshape(c_p, shape=[2, 3])
    print("c_p:\n", c_p)
    print("c_p_reshape:\n", c_p_reshape)

    return None


def variable_demo():
    """
    变量的演示
    :return:
    """
    # 创建变量
    with tf.variable_scope("my_scope"):
        a = tf.Variable(initial_value=50)
        b = tf.Variable(initial_value=40)
    with tf.variable_scope("your_scope"):
        c = tf.add(a, b)
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 开启会话
    with tf.Session() as sess:
        # 运行初始化
        sess.run(init)
        a_value, b_value, c_value = sess.run([a, b, c])
        print("a_value:\n", a_value)
        print("b_value:\n", b_value)
        print("c_value:\n", c_value)

    return None


def linear_regression():
    """
    线性回归案例 自实现
    :return:
    """
    # 1准备数据
    with tf.variable_scope("prepare_data"):
        X = tf.random_normal(shape=[100, 1], name="feature")
        y_true = tf.matmul(X, [[0.8]]) + 0.7

    # 2构建模型
    with tf.variable_scope("create_model"):
        weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Weights")
        bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name="Bias")
        y_predict = tf.matmul(X, weights) + bias

    # 3构造损失函数
    with tf.variable_scope("loss_function"):
        error = tf.reduce_mean(tf.square(y_predict - y_true))  # 均方误差

    # 4优化损失
    with tf.variable_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)  # 梯度下降

    # 显式地初始化变量
    init = tf.global_variables_initializer()

    # 2_收集变量
    tf.summary.scalar("error", error)
    tf.summary.histogram("weights", weights)
    tf.summary.histogram("bias", bias)

    # 3_合并变量
    merged = tf.summary.merge_all()

    # 创建saver对象
    saver = tf.train.Saver()

    # 开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)

        # 1_创建事件文件
        file_writer = tf.summary.FileWriter("./tmp/linear", graph=sess.graph)
        """
        用绝对路径
        tensorboard --logdir="/Users/xxx/Project/TensorFlow_Primer/day1/tmp/linear"                                         
        """
        print("训练前模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))

        # # 开始训练
        # for i in range(500):
        #     sess.run(optimizer)
        #     print("第%d次训练后模型参数为：权重%f，偏置%f，损失为%f" % (i + 1, weights.eval(), bias.eval(), error.eval()))
        #     # 运行合并变量操作
        #     summary = sess.run(merged)
        #     # 将每次迭代后的变量写入事件文件
        #     file_writer.add_summary(summary, i)
        #
        #     # 保存模型
        #     if i % 10 == 0:
        #         saver.save(sess, "./tmp/model/my_linear.ckpt")
        # 加载模型
        if os.path.exists("./tmp/model/checkpoint"):
            saver.restore(sess, "./tmp/model/my_linear.ckpt")

        print("训练后模型参数为：权重%f，偏置%f，损失为%f" % (weights.eval(), bias.eval(), error.eval()))

    return None


# 1）定义命令行参数
tf.app.flags.DEFINE_integer("max_step", 100, "训练模型的步数")
tf.app.flags.DEFINE_string("model_dir", "Unknown", "模型保存的路径+模型名字")

# 2）简化变量名
FLAGS = tf.app.flags.FLAGS

def main(argv):
    print("code start")
    return None

def command_demo():
    """
    命令行参数演示
    :return:
    """
    print("max_step:\n", FLAGS.max_step)
    print("model_dir:\n", FLAGS.model_dir)

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
    # tensor_demo()
    # 代码五： 变量演示
    # variable_demo()
    # 代码六： 线性回归案例
    # linear_regression()
    # 代码七：  命令行参数演示
    # command_demo()
    tf.app.run()