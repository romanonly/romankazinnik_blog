Distributed modeling: all-reduce vs. ring-reduce 

All-reduce original example
Originally published in https://github.com/alexal1/MPI-Gauss-Seidel, I needed to fix some bugs to makes it
to converge.

Install mpi Ubuntu:
sudo apt-get install mpich2

Install mpi MAC OS: http://macappstore.org/mpich2/

Compile and Run:

mpicc seidel_mpi.c -o seidel -lm

mpirun -np 2 ./seidel

Original c-file: seidel_mpi_v0.c

Notice how all-reduce needs to compute all the model parameters, by that requires a full copy of the model on each core. 
All-reduce will take advantage of mmultiple cores by using partial sums at each core and call all-reduce to compute the sums. 

Ring-reduce will compute only a portion of the model parameters. Essentially, it will synchronize model parameters only to the previous core. As a result, for block diagonal matrix ring-reduce will need to store only a small portion of the model.

'No free lunch' for ring-reduce means a longer convergence times and fine-tuning of several more parameters, such as number of
iterations on each core beweetn the syncronization with the previous neighbourt. 

Ring-reduce helps with extremely large models that do not fit into one core/GPU memory.

Matrix size n = 2000
eps = 0.1
Synchronous All-reduce:
1/0 (sync or async)= 1

Asynchronous star-reduce:
1/0 (sync or async)= 0
