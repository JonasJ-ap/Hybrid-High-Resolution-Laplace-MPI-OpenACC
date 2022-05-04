Project Name: High Resolution Laplace
source file: mpiacc_final.c

Introduction:

The project is an extension of the laplace exercise in 33-456 ACP. 
The goal is to make a laplace solver which can solve 10K x 10K problem 
in high speed. The boundary condition is still a linear distribution 
from 0 to 100 celsius degrees.

The code is parallelized by MPI + OpenACC. In other words, the code is optimized to be accelerated
by multiple GPUs. The code implements ghost row approach to resolve communication issue between different PEs.

References: 1. 33-456 ACP lectures
           2. https://www.open-mpi.org/
           3. Nvidia GPU Conference
Compile and run:
1. On bridges2 terminal, type: "interact -p GPU --gres=gpu:v100-32:8 -N 1 -t 30:00 -n 8" to get 8 GPUs

2. Compile the code using: "mpicc -acc -Minfo=accel mpiacc_final.c -o final_mpiacc".

3.  Run with different number of processes(GPUs) and record the time costs, for example: "mpirun -n 4 final_mpiacc" runs the code with 4 processes

4. Note: input the max iterations: a number > 3578 since the laplace solver converges after 3578 iterations
To get the speedup, the project runs: mpirun -n 1 final_mpiacc
                                      mpirun -n 2 final_mpiacc
                                      mpirun -n 3 final_mpiacc
                                      mpirun -n 4 final_mpiacc
                                      mpirun -n 5 final_mpiacc
                                      mpirun -n 6 final_mpiacc
                                      mpirun -n 7 final_mpiacc
                                      mpirun -n 8 final_mpiacc

Output:
The code will output the final temperature matrix in a file named "output_large.txt".
Note: Running with 3,6,or 7 processes may not get full 10K rows since they are not the factor of 10K.