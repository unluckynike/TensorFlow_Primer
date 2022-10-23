'''
@Project ：TensorFlow_Primer 
@File    ：deeplearning_dog.py
@Author  ：hailin
@Date    ：2022/10/23 21:31 
@Info    : 文件读取
'''
import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def picture_read(file_list):
    """
    狗图片案例
    :return:
    """
    # 1构建文件名队列
    file_queue = tf.train.string_input_producer(file_list)

    # 2读取文件与编码
    # 读取阶段
    reader = tf.WholeFileReader()
    # key 文件名 value 一张图片的原始编码格式
    key, value = reader.read(file_queue)
    print("key:\n", key)
    print("value:\n", value)
    # 解码阶段
    image=tf.image.decode_jpeg(value)
    print("image:\n",image)

    # 图像的形状、类型修改
    image_resized=tf.image.resize_images(image,[200,200])
    print("image_resized:\n",image_resized)
    # 确定形状
    image_resized.set_shape(shape=[200,200,3])

    # 3批处理
    image_batch=tf.train.batch([image_resized],batch_size=100,num_threads=1,capacity=32)
    # 批处理前需要确定形状
    print("image_batch:\n",image_batch)

    # 开启会话
    with tf.Session() as sess:
        # 开启线程 否则阻塞
        # 线程协调员
        coord = tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess, coord=coord)

        key_new, value_new ,image_new,image_resized= sess.run([key, value,image,image_resized])
        print("key_new:\n", key_new)
        print("value_new:\n", value_new)
        print("image_new:\n",image_new)
        print("image_resized:\n",image_resized)

        # 回收线程
        coord.request_stop()
        coord.join(threads)

    return None

if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()

    # 构造路径+文件名
    filename = os.listdir("./dog")
    # print(filename)
    # 拼接路径 文件名
    file_list = [os.path.join("./dog/", file) for file in filename]
    # print(file_list)
    picture_read(file_list)
