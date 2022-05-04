/****************************************************************
 * Laplace MPI Template C Version                                         
 *                                                               
 * T is initially 0.0                                            
 * Boundaries are as follows                                     
 *                                                               
 *                T                      4 sub-grids            
 *   0  +-------------------+  0    +-------------------+       
 *      |                   |       |                   |           
 *      |                   |       |-------------------|         
 *      |                   |       |                   |      
 *   T  |                   |  T    |-------------------|             
 *      |                   |       |                   |     
 *      |                   |       |-------------------|            
 *      |                   |       |                   |   
 *   0  +-------------------+ 100   +-------------------+         
 *      0         T       100                                    
 *                                                                 
 * Each PE only has a local subgrid.
 * Each PE works on a sub grid and then sends         
 * its boundaries to neighbors.
 *                                                                 
 *  John Urbanic, PSC 2014
 *
 *******************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <openacc.h>

#define COLUMNS      10000
#define ROWS_GLOBAL  10000      // this is a "global" row count


// communication tags
#define DOWN     100
#define UP       101   

#define MAX_TEMP_ERROR 0.01

void output(int my_pe, int iteration);


int main(int argc, char *argv[]) {

    int i, j;
    int max_iterations;
    int iteration=1;
    double dt;
    struct timeval start_time, stop_time, elapsed_time;

    int        npes;                // number of PEs
    int        my_PE_num;           // my PE number
    double     dt_global=100;       // delta t across all PEs
    // MPI_Status status;              // status returned by MPI calls

    // int target_down, target_up;     // destination of ghost rows
    int sender_down, sender_up;     // source of ghost rows


    // the usual MPI startup routines
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_PE_num);
    
    int ROWS = (ROWS_GLOBAL/npes);
    double Temperature[ROWS+2][COLUMNS+2];
    double Temperature_last[ROWS+2][COLUMNS+2];
    // verify only NPES PEs are being used
    // PE 0 asks for input

    if (my_PE_num == 0) {
        printf("Maximum iterations [100-4000]?\n");
        scanf("%d", &max_iterations);
    }
    // bcast max iterations to other PEs
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    

    // initializa:
    
    double tMin, tMax;  //Local boundary limits
    #pragma acc data copyout(Temperature_last)
    {
    #pragma acc kernels
    for(i = 0; i <= ROWS+1; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }
    

    // Local boundry condition endpoints

    // set right to a linear increase
    #pragma acc kernels
    for(i = 0; i <= ROWS+1; i++) {
        tMin = my_PE_num*100.0/npes;
        tMax = (my_PE_num + 1)*100.0/npes;
        Temperature_last[i][COLUMNS+1] = tMin + ((tMax - tMin)/ROWS)*i;
    }
    // set bottom to linear increase
    #pragma acc kernels
    if (my_PE_num == (npes - 1)) {
        for(j = 0; j <= COLUMNS+1; j++) {
            Temperature_last[ROWS+1][j] = (100.0/COLUMNS)*j;
        }
    }

    }

    // make sure each process has unique gpu device
    int ngpus = acc_get_num_devices(acc_device_nvidia);
	acc_set_device_num(my_PE_num % ngpus, acc_device_nvidia);
    if (my_PE_num == 0) {
        printf("Total device number: %d\n", ngpus);
    }

    printf("My PE %d, My device: %d\n", my_PE_num, my_PE_num % ngpus);

    // Windows setup:
    MPI_Win dwin, uwin;
    if (my_PE_num != (npes - 1)) {
        MPI_Win_create(&Temperature[ROWS][1], sizeof(double)*COLUMNS, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &dwin);
    }
    else {
        MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &dwin);
    }

    if (my_PE_num != 0) {
        MPI_Win_create(&Temperature[1][1], sizeof(double)*COLUMNS, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &uwin);
    }
    else {
        MPI_Win_create(MPI_BOTTOM, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &uwin);
    }

    // timer start
    if (my_PE_num==0) gettimeofday(&start_time,NULL);
   
    
    #pragma acc data copyin(Temperature_last), copyout(Temperature)
    {
    while ( dt_global > MAX_TEMP_ERROR && iteration <= max_iterations ) {


        // main calculation: average my four neighbors
       
        #pragma acc kernels
        for(i = 1; i <= ROWS; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }

        #pragma acc update host(Temperature[ROWS:1][1:COLUMNS], Temperature[1:1][1:COLUMNS])
        // COMMUNICATION PHASE: send and receive ghost rows for next iteration
        
        // downward get:
        MPI_Win_fence(0, dwin);
        if (my_PE_num != 0) {
            sender_down = my_PE_num - 1;
            MPI_Get(&Temperature_last[0][1], COLUMNS, MPI_DOUBLE, sender_down, 0, COLUMNS, MPI_DOUBLE, dwin);
        }
        MPI_Win_fence(0, dwin);

        // upward get:
        MPI_Win_fence(0, uwin);
        if (my_PE_num != (npes - 1)) {
            sender_up = my_PE_num + 1;
            MPI_Get(&Temperature_last[ROWS+1][1], COLUMNS, MPI_DOUBLE, sender_up, 0, COLUMNS, MPI_DOUBLE, uwin);
        }
        MPI_Win_fence(0, uwin);

        dt = 0.0;

        #pragma acc update device(Temperature_last[0:1][1:COLUMNS], Temperature_last[ROWS+1:1][1:COLUMNS])
        
         #pragma acc data copy(dt)
        {
        #pragma acc kernels loop reduction (max:dt) 
        for(i = 1; i <= ROWS; i++){
            for(j = 1; j <= COLUMNS; j++){
	        dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	        Temperature_last[i][j] = Temperature[i][j];
            }
        }
        }

        // find global dt

        MPI_Reduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dt_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // periodically print test values - only for PE in lower corner
        if((iteration % 100) == 0) {
            if (my_PE_num == npes-1){
                #pragma acc update host(Temperature[ROWS-5:6][COLUMNS-5:6])
                
                int i;

                printf("---------- Iteration number: %d ------------\n", iteration);

                // output global coordinates so user doesn't have to understand decomposition
                for(i = 5; i >= 0; i--) {
                printf("[%d,%d]: %5.2f  ", ROWS_GLOBAL-i, COLUMNS-i, Temperature[ROWS-i][COLUMNS-i]);
                }
                printf("\n");

	    }
        }

	iteration++;
    }
    }

    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);

    // PE 0 finish timing and output values
    if (my_PE_num==0){
        gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time);

	printf("\nMax error at iteration %d was %f\n", iteration-1, dt_global);
	printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    }

    // output
    FILE* fp;
    char filename[50];
    sprintf(filename,"output_large.txt");
    for (int pe = 0; pe < npes; pe++){
        if (my_PE_num==pe){
            fp = fopen(filename, "a");
            for(int y = 1; y <= ROWS; y++){
                for(int x = 1; x <= COLUMNS; x ++){
                    fprintf(fp, "%f ",Temperature[y][x]);
                    }
                fprintf(fp,"\n");
            }
            fflush(fp);
            fclose(fp);
        }
    MPI_Barrier(MPI_COMM_WORLD);
    }


    MPI_Win_free(&dwin);
    MPI_Win_free(&uwin);
    MPI_Finalize();
}
