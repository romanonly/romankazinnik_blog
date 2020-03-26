#
# How to use:
# must import cpu, gpu into __main__ space
# from lstm_package.lstm_cuda.lstm_gpu_cuda_imbd import test_cpu_vs_gpu , cpu, gpu
# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import timeit
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

import tensorflow as tf
import sys
import timeit, functools


class ModelParams(object):
    def __init__(self):
        self.model = None
        #        """cut texts after this number of words (among top max_features most common words)"""
        self.max_features = 1000  # 20000
        self.maxlen = 80
        self.batch_size = 512
        self.num_embed = 120  # 128
        self.num_train = 1000  # x_train.shape[0]
        self.num_test = 100
        self.results = None  # nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None


class LstmCpuGpuCudaImdb(object):

    @staticmethod
    def data(t):
        print("Loading data...")
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=t.max_features)
        print(len(x_train), "train sequences")
        print(len(x_test), "test sequences")
        print("Pad sequences (samples x time)")
        x_train = sequence.pad_sequences(x_train, maxlen=t.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=t.maxlen)
        print("x_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def data_pickle(t):
        import pickle, sys
        print('Loading data...')
        max_features = t.max_features  # 1000
        maxlen = t.maxlen  # 100

        try:
            from tensorflow.keras.datasets import imdb
            from tensorflow.keras.preprocessing import sequence

            print(" reading data from imdb.load_data")
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

            dt_string = "imdb_cuda_lstm_test.pkl"
            with open(dt_string, 'wb') as f:
                pickle.dump([x_train, y_train, x_test, y_test], f)
        except:
            print(" except unexpected error:", sys.exc_info()[0])
            print(" reading data from", dt_string)
            import _pickle as cPickle
            objects = []
            with (open(dt_string, "rb")) as openfile:
                while True:
                    try:
                        objects.append(cPickle.load(openfile))
                    except EOFError:
                        break
            assert len(objects) == 1
            d = objects[0]
            x_train, y_train, x_test, y_test = d[0], d[1], d[2], d[3]

        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print('Pad sequences (samples x time)')
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def make_model(t, is_cuda):
        """ model will be saved in t """

        print("\n Build model...")
        t.model = Sequential()
        t.model.add(Embedding(t.max_features, t.num_embed))
        try:
            if is_cuda:
                print("Model add layer CuDNNLSTM...")
                from tensorflow.compat.v1.keras.layers import CuDNNLSTM

                t.model.add(CuDNNLSTM(t.num_embed))
            else:
                # GPU must have recurrent_dropout=0.0
                t.model.add(LSTM(t.num_embed, dropout=0.2, recurrent_dropout=0.0))
        except:
            print("Unexpected error:", sys.exc_info()[0])
        t.model.add(Dense(1, activation="sigmoid"))
        t.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    @staticmethod
    def eval_model(t):
        print("Evaluate model")
        score, acc = t.model.evaluate(
            t.x_test[-t.num_test :, :],
            t.y_test[-t.num_test :],
            batch_size=t.batch_size,
            verbose=0,
        )
        print("Test score:", score, "Test accuracy:", acc)
        return score, acc

    @staticmethod
    def print_results(
        nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu, num_embed, num_train, num_test
    ):
        print(
            "\n TENSOR \n cpu_vs_gpu speedup=",
            (cpu_time / gpu_time),
            "( cpu,gpu=",
            cpu_time,
            gpu_time,
            ")",
        )
        s = "LSTM:\nnum_embed={} num_train={} num_test={}".format(
            num_embed, num_train, num_test
        )
        print(s)
        print(
            "LSTM NO-CUDA: cpu VS gpu speedup=",
            (nocuda_cpu / nocuda_gpu),
            "( cpu,gpu=",
            nocuda_cpu,
            nocuda_gpu,
            ")",
        )
        print(
            "LSTM CUDA: cpu VS gpu speedup=",
            (cuda_cpu / cuda_gpu),
            "( cpu,gpu=",
            cuda_cpu,
            cuda_gpu,
            ")",
        )
        print("LSTM CUDA VS No-CUDA speedup=", (nocuda_cpu / cuda_gpu))

    @staticmethod
    def cpu_run_model(t):
        with tf.device("/cpu:0"):
            t.model.fit(
                t.x_train[: t.num_train, :],
                t.y_train[: t.num_train],
                batch_size=t.batch_size,
                epochs=1,
                validation_data=(t.x_test[: t.num_test, :], t.y_test[: t.num_test]),
            )

    @staticmethod
    def gpu_run_model(t):
        try:
            with tf.device("/device:GPU:0"):
                t.model.fit(
                    t.x_train[: t.num_train, :],
                    t.y_train[: t.num_train],
                    batch_size=t.batch_size,
                    epochs=1,
                    validation_data=(t.x_test[: t.num_test, :], t.y_test[: t.num_test]),
                )
        except:
            print(" LstmCpuGpuCudaImdb.gpu_run_model except unexpected error:", sys.exc_info()[0])
            return sys.exc_info()[0]

    @staticmethod
    def test_cpu_vs_gpu_vs_cuda(t, num_repeat=1):
        print("Build model...")
        print("\n Run LSTM ...\n")
        print("Time (s) to .... ")
        print("CPU (s):")

        # cpu_time = timeit.timeit('LstmCpuGpuCudaImdb.cpu_run_model()', number=num_repeat, setup="from __main__ import LstmCpuGpuCudaImdb")
        def foo1(t):
            LstmCpuGpuCudaImdb.cpu_run_model(t)

        t1 = timeit.Timer(functools.partial(foo1, t))
        cpu_time = t1.timeit(number=num_repeat)
        print("cpu_time=", cpu_time)
        LstmCpuGpuCudaImdb.eval_model(t)

        print("GPU (s):")
        # gpu_time = timeit.timeit('LstmCpuGpuCudaImdb.gpu_run_model()', number=num_repeat, setup="from __main__ import LstmCpuGpuCudaImdb")
        def foo2(t):
            LstmCpuGpuCudaImdb.gpu_run_model(t)

        t2 = timeit.Timer(functools.partial(foo2, t))
        gpu_time = t2.timeit(number=num_repeat)
        print("gpu_time=", gpu_time)
        LstmCpuGpuCudaImdb.eval_model(t)
        print("\n\n GPU speedup over CPU: {}x".format((cpu_time / gpu_time)))
        return cpu_time, gpu_time

    @staticmethod
    def lstm_cpu_gpu_cuda(t):
        nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = -1, -1, -1, -1

        LstmCpuGpuCudaImdb.make_model(t, is_cuda=False)
        #        """warmup"""
        LstmCpuGpuCudaImdb.cpu_run_model(t), LstmCpuGpuCudaImdb.gpu_run_model(t)
        nocuda_cpu, nocuda_gpu = LstmCpuGpuCudaImdb.test_cpu_vs_gpu_vs_cuda(
            t, num_repeat=1
        )
        try:
            LstmCpuGpuCudaImdb.make_model(t, is_cuda=True)
            #           """warmup"""
            LstmCpuGpuCudaImdb.cpu_run_model(t), LstmCpuGpuCudaImdb.gpu_run_model(t)
            cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.test_cpu_vs_gpu_vs_cuda(
                t, num_repeat=1
            )
        except:
            print("\n\n lstm_cpu_gpu_cuda unexpected error:\n", sys.exc_info()[0])

        return nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
