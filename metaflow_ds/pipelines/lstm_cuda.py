from metaflow import FlowSpec, step, Parameter

# current should be '02-lstm-cuda'
import os
import sys
print(os.getcwd())
#Note the single dot as opposed to the two-dot.
sys.path.append(".")

from metaflow_ds.components.cpu_gpu import SpeedupCpuGpuCuda

from metaflow_ds.components.lstm_cpu_gpu_cuda_idbm import (
    LstmCpuGpuCudaImdb,
    ModelParams,
)


def script_path(filename):
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class LSTMGPUCudaFlow(FlowSpec):
    """
    A flow to generate LSTM GPU and CUDA speedup statistics.

    Run this:

    python metaflow_ds/pipelines/lstm_cuda.py run --size 1000 --lstmruns 1
    jupyter-notebook metaflow_ds/pipelines/lstm_cuda.ipynb


    1) Ingests input parametsr: LSTM size, data size
    2) Fan-out over LSTM size using Metaflow foreach.
    3) Compute stats for each genre.
    4) Save a dictionary of LSTMsize specific statistics.

    """


    size = Parameter(
        "size", help="Number of data points.", type=int, default=250,
    )
    num_lstm_experiments = Parameter(
        "lstmruns", help="Number of lstm runs: 128, 256,512 etc.", type=int, default=1,
    )

    @step
    def start(self):
        """
        The start step:
        1) runs TF simple speedup test.
        2) Create lstm experiments.
        3) Launches parallel statistics computation for each LSTM experiment.

        """
        import numpy as np

        # Init your classes, etc.

        self.cpu_time, self.gpu_time = SpeedupCpuGpuCuda.test_cpu_vs_gpu()

        # 128, 256, 512, 1024
        self.genres = 128 * np.power(2, np.array(range(0, self.num_lstm_experiments)))
        self.genres = list(self.genres)

        # compute statistics for each genre. The 'foreach'
        # keyword argument allows us to compute the statistics for each genre in
        # parallel (i.e. a fan-out).
        self.next(self.compute_statistics, foreach='genres')

    @step
    def compute_statistics(self):
        """
        Compute statistics for a single genre.

        """
        # The genre currently being processed is a class property called
        # 'input'.
        self.genre = self.input
        print("Computing statistics for LSTM size=%s" % self.genre)

        #    """ similar CPU and GPU, CUDA"""
        t = ModelParams()
        t.x_train, t.y_train, t.x_test, t.y_test = LstmCpuGpuCudaImdb.data_pickle(t)
        t.batch_size = 512  # larger batch size will result even higher speedups GPU/CUDA
        t.num_embed, t.num_train, t.num_test = self.genre, 250, 25

        nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = -1, -1, -1, -1

        #nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu = LstmCpuGpuCudaImdb.lstm_cpu_gpu_cuda(t)

        # t.results = nocuda_cpu, nocuda_gpu, cuda_cpu, cuda_gpu
        # t01 = copy.copy(t)

        # Get some statistics on the gross box office for these titles.
        self.cpu = nocuda_cpu
        self.gpu = nocuda_gpu
        self.cpu_cuda = cuda_cpu
        self.gpu_cuda = cuda_gpu

        # Join the results from other genres.
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join our parallel branches and merge results into a dictionary.

        """
        self.model_stats = {
            inp.genre: {
                "cpu": inp.cpu,
                "gpu": inp.gpu,
                "cpu_cuda": inp.cpu_cuda,
                "gpu_cuda": inp.gpu_cuda,
            }
            for inp in inputs
        }
        self.model_stats['self_data'] = {
#            "x_train_shape": self.x_train.shape,
#            "x_test_shape": self.x_test.shape,
        }

        self.next(self.end)


    @step
    def end(self):
        """
        End the flow.

        """
        pass


if __name__ == '__main__':
    LSTMGPUCudaFlow()
