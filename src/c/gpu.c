/**
 * @file main.c
 * @brief This file contains the source code of the application to parallelise.
 * @details This application is a classic heat spread simulation.
 * @author Ludovic Capelli
 **/

#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <inttypes.h>
//#include <math.h>
#include <accelmath.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

#include "util.h"

#define DEBUG_TIMING_OUTPUT 1

/**
 * @argv[0] Name of the program
 * @argv[1] path to the dataset to load
 **/
int main(int argc, char* argv[])
{
	MPI_Init(NULL, NULL);

	/////////////////////////////////////////////////////
	// -- PREPARATION 1: COLLECT USEFUL INFORMATION -- //
	/////////////////////////////////////////////////////
	// Ranks for convenience so that we don't throw raw values all over the code
	const int MASTER_PROCESS_RANK = 0;

	// The rank of the MPI process in charge of this instance
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Number of MPI processes in total, commonly called "comm_size" for "communicator size".
	int comm_size;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	/// Rank of the first MPI process
	const int FIRST_PROCESS_RANK = 0;
	/// Rank of the last MPI process
	const int LAST_PROCESS_RANK = comm_size - 1;

	// Rank of my up neighbour if any
	int up_neighbour_rank = (my_rank == FIRST_PROCESS_RANK) ? MPI_PROC_NULL : my_rank - 1;

	// Rank of my down neighbour if any
	int down_neighbour_rank = (my_rank == LAST_PROCESS_RANK) ? MPI_PROC_NULL : my_rank + 1;

	//report_placement();

	////////////////////////////////////////////////////////////////////
	// -- PREPARATION 2: INITIALISE TEMPERATURES ON MASTER PROCESS -- //
	////////////////////////////////////////////////////////////////////

	/// Array that will contain my part chunk. It will include the 2 ghost rows (1 up, 1 down)
	double temperatures[ROWS_PER_MPI_PROCESS+2][COLUMNS_PER_MPI_PROCESS];
	/// Temperatures from the previous iteration, same dimensions as the array above.
	double temperatures_last[ROWS_PER_MPI_PROCESS+2][COLUMNS_PER_MPI_PROCESS];
	/// On master process only: contains all temperatures read from input file.
	double all_temperatures[ROWS][COLUMNS];

	// The master MPI process will read a chunk from the file, send it to the corresponding MPI process and repeat until all chunks are read.
	if(my_rank == MASTER_PROCESS_RANK)
	{
		initialise_temperatures(all_temperatures);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	///////////////////////////////////////////
	//     ^                                 //
	//    / \                                //
	//   / | \    CODE FROM HERE IS TIMED    //
	//  /  o  \                              //
	// /_______\                             //
	///////////////////////////////////////////

	////////////////////////////////////////////////////////
	// -- TASK 1: DISTRIBUTE DATA TO ALL MPI PROCESSES -- //
	////////////////////////////////////////////////////////
	double total_time_so_far = 0.0;
	double total_time_so_far_2 = 0.0;
	double start_time = MPI_Wtime();
	double start_time_2 = MPI_Wtime();

	if(my_rank == MASTER_PROCESS_RANK)
	{
		for(int i = 0; i < comm_size; i++)
		{
			// Is the i'th chunk meant for me, the master MPI process?
			if(i != my_rank)
			{
				MPI_Request request;
				// No, so send the corresponding chunk to that MPI process.
				MPI_Isend(&all_temperatures[i * ROWS_PER_MPI_PROCESS][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
				MPI_Request_free(&request);
			}
		}

		// Yes, let's copy it straight for the array in which we read the file into.
		for(int j = 1; j <= ROWS_PER_MPI_PROCESS; j++)
		{
			for(int k = 0; k < COLUMNS_PER_MPI_PROCESS; k++)
			{
				temperatures_last[j][k] = all_temperatures[j-1][k];
			}
		}
	}
	else
	{
		// Receive my chunk.
		MPI_Recv(&temperatures_last[1][0], ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Copy the temperatures into the current iteration temperature as well
	for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
	{
		for(int j = 0; j < COLUMNS_PER_MPI_PROCESS; j++)
		{
			temperatures[i][j] = temperatures_last[i][j];
		}
	}

	if(my_rank == MASTER_PROCESS_RANK)
	{
		printf("Data acquisition complete.\n");
	}

	// Wait for everybody to receive their part before we can start processing
	MPI_Barrier(MPI_COMM_WORLD);

	/////////////////////////////
	// TASK 2: DATA PROCESSING //
	/////////////////////////////
	int iteration_count = 0;
	double global_temperature_change; /// Maximum temperature change observed across all MPI processes
	double my_temperature_change; /// Maximum temperature change for us
	double snapshot[ROWS][COLUMNS]; /// The last snapshot made

	acc_set_device_num( my_rank, acc_device_nvidia );

	//MPI_Request snapshot_request = MPI_REQUEST_NULL;
	const size_t buffer_size = ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS;
	//void * snapshot_buffer = malloc(buffer_size * sizeof(double));

	total_time_so_far_2 = MPI_Wtime() - start_time_2;
	if(my_rank == MASTER_PROCESS_RANK)
		printf("Init took %.2f.\n", total_time_so_far_2);

	#pragma acc data copyin(temperatures_last, temperatures)
	while(total_time_so_far < MAX_TIME)
	{
		// ////////////////////////////////////////
		// -- SUBTASK 1: EXCHANGE GHOST CELLS -- //
		// ////////////////////////////////////////
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();

		#pragma acc update host(temperatures[1:1][0:COLUMNS_PER_MPI_PROCESS], temperatures[ROWS_PER_MPI_PROCESS:1][0:COLUMNS_PER_MPI_PROCESS])

		// Send data to up neighbour for its ghost cells. If my up_neighbour_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
		MPI_Ssend(&temperatures[1][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, up_neighbour_rank, 0, MPI_COMM_WORLD);

		// Receive data from down neighbour to fill our ghost cells. If my down_neighbour_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
		MPI_Recv(&temperatures_last[ROWS_PER_MPI_PROCESS+1][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, down_neighbour_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Send data to down neighbour for its ghost cells. If my down_neighbour_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
		MPI_Ssend(&temperatures[ROWS_PER_MPI_PROCESS][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, down_neighbour_rank, 0, MPI_COMM_WORLD);

		// Receive data from up neighbour to fill our ghost cells. If my up_neighbour_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
		MPI_Recv(&temperatures_last[0][0], COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE, up_neighbour_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		#pragma acc update device(temperatures_last[ROWS_PER_MPI_PROCESS+1:1][0:COLUMNS_PER_MPI_PROCESS], temperatures_last[0:1][0:COLUMNS_PER_MPI_PROCESS])

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 1 took %.18f.\n", total_time_so_far_2);
		}

		/////////////////////////////////////////////
		// -- SUBTASK 2: PROPAGATE TEMPERATURES -- //
		/////////////////////////////////////////////
		// Define temp reduction variables
		double temp1 = 0, temp2 = 0, temp3 = 0;
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();

		#pragma acc kernels
		{
			#pragma acc loop independent
			for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
			{
				// Process the cell at the first column, which has no left neighbour
				if(temperatures[i][0] != MAX_TEMPERATURE)
				{
					temperatures[i][0] = (
						temperatures_last[i-1][0] +
						temperatures_last[i+1][0] +
						temperatures_last[i  ][1]
						) / 3.0;
				}
				temp1 = fmax(fabs(temperatures[i][0] - temperatures_last[i][0]), temp1);
			}

			#pragma acc loop independent tile(32,32)
			for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
			{
				// Process all cells between the first and last columns excluded, which each has both left and right neighbours
				for(int j = 1; j < COLUMNS_PER_MPI_PROCESS - 1; j++)
				{
					if(temperatures[i][j] != MAX_TEMPERATURE)
					{
						temperatures[i][j] = 0.25 * (
							temperatures_last[i-1][j  ] +
							temperatures_last[i+1][j  ] +
							temperatures_last[i  ][j-1] +
							temperatures_last[i  ][j+1]
							);
					}
					temp2 = fmax(fabs(temperatures[i][j] - temperatures_last[i][j]), temp2);
				}
			}

			#pragma acc loop independent
			for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
			{
				// Process the cell at the last column, which has no right neighbour
				if(temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] != MAX_TEMPERATURE)
				{
					temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] = (
						temperatures_last[i-1][COLUMNS_PER_MPI_PROCESS - 1] +
						temperatures_last[i+1][COLUMNS_PER_MPI_PROCESS - 1] +
						temperatures_last[i  ][COLUMNS_PER_MPI_PROCESS - 2]
						) / 3.0;
					temp3 = fmax(fabs(temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] - temperatures_last[i][COLUMNS_PER_MPI_PROCESS - 1]), temp3);
				}
			}
		}

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 2 took %.18f.\n", total_time_so_far_2);
		}

		///////////////////////////////////////////////////////
		// -- SUBTASK 3: CALCULATE MAX TEMPERATURE CHANGE -- //
		///////////////////////////////////////////////////////
		// only need to reduce the values from the 3 subprocesses
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();
		
		my_temperature_change = fmax(fmax(temp1, temp2), temp3);

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 3 took %.18f.\n", total_time_so_far_2);
		}

		//////////////////////////////////////////////////////////
		// -- SUBTASK 4: FIND MAX TEMPERATURE CHANGE OVERALL -- //
		//////////////////////////////////////////////////////////
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();

		MPI_Reduce(&my_temperature_change, &global_temperature_change, 1, MPI_DOUBLE, MPI_MAX, MASTER_PROCESS_RANK, MPI_COMM_WORLD);

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 4 took %.18f.\n", total_time_so_far_2);
		}

		//////////////////////////////////////////////////
		// -- SUBTASK 5: UPDATE LAST ITERATION ARRAY -- //
		//////////////////////////////////////////////////
		/*
		237: compute region reached 1226 times
        245: kernel launched 1226 times
            grid: [65535]  block: [128]
             device time(us): total=737,846 max=605 min=599 avg=601
            elapsed time(us): total=760,984 max=637 min=618 avg=620
			*/
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();

		#pragma acc kernels loop independent collapse(2) async(1)
		for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
		{
			for(int j = 0; j < COLUMNS_PER_MPI_PROCESS; j++)
			{
				temperatures_last[i][j] = temperatures[i][j];
			}
		}

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 5 took %.18f.\n", total_time_so_far_2);
		}

		///////////////////////////////////
		// -- SUBTASK 6: GET SNAPSHOT -- //
		///////////////////////////////////
		/*
		MPI_Igather(void* buffer_send,
                int count_send,
                MPI_Datatype datatype_send,
                void* buffer_recv,
                int count_recv,
                MPI_Datatype datatype_recv,
                int root,
                MPI_Comm communicator,
                MPI_Request* request);
				*/
		if(DEBUG_TIMING_OUTPUT)
			start_time_2 = MPI_Wtime();
		
		if(0)
		if(iteration_count % SNAPSHOT_INTERVAL == 0)
		{	
			if(my_rank == MASTER_PROCESS_RANK)
			{
				printf("Iteration %d: %.18f\n", iteration_count, global_temperature_change);
			}
			//if(snapshot_request != MPI_REQUEST_NULL) MPI_Wait(&snapshot_request, MPI_STATUS_IGNORE);
			#pragma acc update host(temperatures[1:ROWS_PER_MPI_PROCESS][0:COLUMNS_PER_MPI_PROCESS])
			//memcpy(snapshot_buffer, temperatures, buffer_size * sizeof(double));
			//MPI_Igather(snapshot_buffer, buffer_size, MPI_DOUBLE, snapshot, buffer_size, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_COMM_WORLD, &snapshot_request);
			MPI_Gather(&temperatures[1][0], buffer_size, MPI_DOUBLE, snapshot, buffer_size, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_COMM_WORLD);
		}

		if(DEBUG_TIMING_OUTPUT) {
			total_time_so_far_2 = MPI_Wtime() - start_time_2;
			if(my_rank == MASTER_PROCESS_RANK)
				printf("Subtask 6 took %.18f.\n", total_time_so_far_2);
		}

		// Calculate the total time spent processing
		if(my_rank == MASTER_PROCESS_RANK)
		{
			total_time_so_far = MPI_Wtime() - start_time;
		}

		// Send total timer to everybody so they too can exit the loop if more than the allowed runtime has elapsed already
		MPI_Bcast(&total_time_so_far, 1, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_COMM_WORLD);

		#pragma acc wait(1)

		// Update the iteration number
		iteration_count++;
	}

	//if(snapshot_request != MPI_REQUEST_NULL)
	//	MPI_Wait(&snapshot_request, MPI_STATUS_IGNORE);

	///////////////////////////////////////////////
	//     ^                                     //
	//    / \                                    //
	//   / | \    CODE FROM HERE IS NOT TIMED    //
	//  /  o  \                                  //
	// /_______\                                 //
	///////////////////////////////////////////////

	/////////////////////////////////////////
	// -- FINALISATION 2: PRINT SUMMARY -- //
	/////////////////////////////////////////
	if(my_rank == MASTER_PROCESS_RANK)
	{
		printf("The program took %.2f seconds in total and executed %d iterations.\n", total_time_so_far, iteration_count);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
