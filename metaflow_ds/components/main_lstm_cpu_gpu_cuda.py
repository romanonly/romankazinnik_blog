"""
#Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

**Notes**

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

"""
# from __future__ import print_function
import tensorflow as tf
from tensorflow.python.client import device_lib
import copy
from datetime import datetime
import pickle


from cpu_gpu import SpeedupCpuGpuCuda

print(
    "__file__={0:<35} | __name__={1:<20} | __package__={2:<20}".format(
        __file__, __name__, str(__package__)
    )
)

from lstm_cpu_gpu_cuda_idbm import (
    LstmCpuGpuCudaImdb,
    ModelParams,
)

print(
    "__file__={0:<35} | __name__={1:<20} | __package__={2:<20}".format(
        __file__, __name__, str(__package__)
    )
)

if __name__ == "__main__":

    print("Show System RAM Memory:\n\n")
    #    """    !cat / proc / meminfo | egrep """
    #    """    "MemTotal*"  """
    print("\n\nShow Devices:\n\n" + str(device_lib.list_local_devices()))
    print("\n\n tf version = ", tf.__version__)

    cpu_time, gpu_time = SpeedupCpuGpuCuda.test_cpu_vs_gpu()

    t = ModelParams()
    t.batch_size = 512  # larger batch size will result even higher speedups GPU/CUDA
    t.num_embed, t.num_train, t.num_test = 12, 250, 25

    t.x_train, t.y_train, t.x_test, t.y_test = LstmCpuGpuCudaImdb.data(t)

    #    """ similar CPU and GPU, CUDA"""
    nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.lstm_cpu_gpu_cuda(t)
    t.results = nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
    t01 = copy.copy(t)

    #    """ some speedups in GPU, CUDA"""
    t.num_embed, t.num_train, t.num_test = 128, 250, 25
    nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.lstm_cpu_gpu_cuda(t)
    t.results = nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
    t0 = copy.copy(t)

    #    """ x30 GPU, x90 CUDA"""
    t.num_embed, t.num_train, t.num_test = 256, 2500, 25
    nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.lstm_cpu_gpu_cuda(t)
    t.results = nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
    t1 = copy.copy(t)

    #    """ x30 GPU, x90 CUDA"""
    t.num_embed, t.num_train, t.num_test = 512, 2500, 25
    nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.lstm_cpu_gpu_cuda(t)
    t.results = nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
    t2 = copy.copy(t)

    LstmCpuGpuCudaImdb.print_results(
        t01.results[0],
        t01.results[1],
        t01.results[2],
        t01.results[3],
        t01.num_embed,
        t01.num_train,
        t01.num_test,
    )
    LstmCpuGpuCudaImdb.print_results(
        t0.results[0],
        t0.results[1],
        t0.results[2],
        t0.results[3],
        t0.num_embed,
        t0.num_train,
        t0.num_test,
    )
    LstmCpuGpuCudaImdb.print_results(
        t1.results[0],
        t1.results[1],
        t1.results[2],
        t1.results[3],
        t1.num_embed,
        t1.num_train,
        t1.num_test,
    )
    LstmCpuGpuCudaImdb.print_results(
        t2.results[0],
        t2.results[1],
        t2.results[2],
        t2.results[3],
        t2.num_embed,
        t2.num_train,
        t2.num_test,
    )

    # save results
    speedup = nocuda_cpu / cuda_gpu
    now = datetime.now()
    print("now =", now)
    dt_string = now.strftime(
        "lstm_cpu_gpu_cuda_speedup_{}_dmY_HMS_%d-%m-%Y-%H_%M_%S.pkl".format(
            int(speedup)
        )
    )
    print("File name =", dt_string)
    """ can't pickle _thread.RLock objects """
    t01.model = t0.model = t1.model = t2.model = None
    with open(dt_string, "wb") as f:
        pickle.dump([t01, t0, t1, t2], f)
