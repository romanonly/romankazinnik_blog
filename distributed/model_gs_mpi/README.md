Distributed modeling: synchronous vs. asynchronous 

Sync example 
Originally published in https://github.com/alexal1/MPI-Gauss-Seidel, I needed to fix some bugs to makes it
to converge.

Install mpi Ubuntu:
sudo apt-get install mpich2

Install mpi MAC OS: http://macappstore.org/mpich2/

Compile and Run:

mpicc seidel_mpi.c -o seidel -lm

mpirun -np 2 ./seidel

Notice how the synchronous convergence time is impeded by 5-second wait that I added on the second core. It illustrates that 
the sync modeling would be as fast as its slowest core. 