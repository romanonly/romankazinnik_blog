#
# How to use:
# must import cpu, gpu into __main__ space
# from lstm_package.lstm_cuda.lstm_gpu_cuda_imbd import test_cpu_vs_gpu , cpu, gpu
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import timeit
import tensorflow as tf
import sys

class SpeedupCpuGpuCuda(object):
    @staticmethod
    def cpu():
        with tf.device("/cpu:0"):
            random_image_cpu = tf.random.normal((100, 100, 10, 3))
            net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
            return tf.math.reduce_sum(net_cpu)

    @staticmethod
    def gpu():
        try:
            with tf.device("/device:GPU:0"):
                random_image_gpu = tf.random.normal((100, 100, 10, 3))
                net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
                return tf.math.reduce_sum(net_gpu)
        except:
            print(" SpeedupCpuGpuCuda.gpu except unexpected error:", sys.exc_info()[0])
            return sys.exc_info()[0]

    @staticmethod
    def test_cpu_vs_gpu():
        """ # CPU vs GPU: tensorflow multiplication """
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            print(
                "\n\nThis error most likely means that this notebook is not "
                "configured to use a GPU.  Change this in Notebook Settings via the "
                "command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n"
            )
        #    """raise SystemError('GPU device not found')"""
        #    """ We run each op once to warm up; see: https://stackoverflow.com/a/45067900"""
        SpeedupCpuGpuCuda.cpu()
        SpeedupCpuGpuCuda.gpu()
        #    """Run the op several times."""
        print(
            "Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images "
            "(batch x height x width x channel). Sum of ten runs."
        )
        print("CPU (s):")
        cpu_time = timeit.timeit(
            "SpeedupCpuGpuCuda.cpu()",
            number=10,
            setup="from __main__ import SpeedupCpuGpuCuda",
        )
        print(cpu_time)
        print("GPU (s):")
        gpu_time = timeit.timeit(
            "SpeedupCpuGpuCuda.gpu()",
            number=10,
            setup="from __main__ import SpeedupCpuGpuCuda",
        )
        print(gpu_time)
        print("GPU speedup over CPU: {}x".format((cpu_time / gpu_time)))
        return cpu_time, gpu_time
