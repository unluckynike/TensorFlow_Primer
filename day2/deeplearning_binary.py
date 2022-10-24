'''
@Project ：TensorFlow_Primer 
@File    ：deeplearning_binary.py
@Author  ：hailin
@Date    ：2022/10/23 22:59 
@Info    : CIFAR10  读取二进制文件
'''
import tensorflow.compat.v1 as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Cifar:
    # 初始化操作
    def __init__(self):
        # 设置图像大小
        self.width = 32
        self.height = 32
        self.channel = 3
        # 设置图像字节数
        self.image = self.height * self.width * self.channel
        self.label = 1
        self.sample = self.image + self.label

    def read_and_decode(self, file_list):
        # 1构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)

        # 2读取与解码
        # 读取
        reader = tf.FixedLengthRecordReader(self.sample)
        key, value = reader.read(file_queue)
        print("key:\n", key)
        print("value:\n", value)

        # 解码
        image_decoded = tf.decode_raw(value, tf.uint8)
        print("decode:\n", image_decoded)

        # 切片
        label = tf.slice(image_decoded, [0], [self.label])
        image = tf.slice(image_decoded, [self.label], [self.image])
        print("label:\n", label)
        print("image:\n", image)

        # 调整图片形状
        image_reshaped = tf.reshape(image, [self.channel, self.height, self.width])
        print("image_reshaped:\n", image_reshaped)

        # 三维数组的转置
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0])
        print("image_transposed:\n", image_transposed)

        # 3批处理
        image_batch, label_batch = tf.train.batch([image_transposed, label], batch_size=100, num_threads=2,
                                                  capacity=100)

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            key_new, value_new, decode_new, label_new, image_new, image_reshaped_new, image_batch_new = sess.run(
                [key, value, image_decoded, label, image, image_reshaped, image_batch])
            print("key_new:\n", key_new)
            print("value_new:\n", value_new)
            print("decode_new:\n", decode_new)
            print("label_new:\n", label_new)
            print("image_new:\n", image_new)
            print("image_reshaped_new:\n", image_reshaped_new)
            print("image_batch_new:\n", image_batch_new)

            coord.request_stop()
            coord.join(threads)

        return None


    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将样本的特征值和目标值一起写入tfrecords文件
        :param image:
        :param label:
        :return:
        """
        with tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                # print("tfrecords_image:\n", image)
                # print("tfrecords_label:\n", label)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                }))
                # example.SerializeToString()
                # 将序列化后的example写入文件
                writer.write(example.SerializeToString())

        return None


if __name__ == '__main__':
    print(tf.__version__)
    tf.compat.v1.disable_eager_execution()

    file_name = os.listdir("./cifar-10-batches-bin")
    print("filename:\n", file_name)

    # 构造文件名路径列表
    # 如果文件名的后三个字符为 bin  则拼接
    file_list = [os.path.join("./cifar-10-batches-bin/", file) for file in file_name if file[-3:] == "bin"]
    print("file_list:\n", file_list)

    # 实例化
    cifar = Cifar()
    cifar.read_and_decode(file_list)  # 传入文件列表
