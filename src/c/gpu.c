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
	double temperatures[ROWS][COLUMNS];
	/// Temperatures from the previous iteration, same dimensions as the array above.
	double temperatures_last[ROWS][COLUMNS];
	/// On master process only: contains all temperatures read from input file.
//	double all_temperatures[ROWS][COLUMNS];

	// The master MPI process will read a chunk from the file, send it to the corresponding MPI process and repeat until all chunks are read.
	if(my_rank == MASTER_PROCESS_RANK)
	{
		initialise_temperatures(temperatures);
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
	double start_time = MPI_Wtime();

	for(int j = 0; j < ROWS; j++)
	{
		for(int k = 0; k < COLUMNS; k++)
		{
			temperatures_last[j][k] = temperatures[j][k];
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
	if(my_rank != MASTER_PROCESS_RANK) return 0;
printf("here-1\n");

	#pragma acc data copyin(temperatures_last, temperatures)
	while(total_time_so_far < MAX_TIME)
	{
		// ////////////////////////////////////////
		// -- SUBTASK 1: EXCHANGE GHOST CELLS -- //
		// ////////////////////////////////////////
		// N/A for this implemenation

		/////////////////////////////////////////////
		// -- SUBTASK 2: PROPAGATE TEMPERATURES -- //
		/////////////////////////////////////////////
		// Define temp reduction variables
		double temp1 = 0, temp2 = 0, temp3 = 0;
printf("here1\n");
		#pragma acc kernels
		{
			/*
			#pragma acc loop independent
			for(int i = 0; i < ROWS; i++)
			{
				// Process the cell at the first column, which has no left neighbour
				if(temperatures[i][0] != MAX_TEMPERATURE)
				{
					if(i==0)
						temperatures[i][0] = (  temperatures_last[i+1][0] + temperatures_last[i  ][1] ) / 2.0;
					else if(i==ROWS-1)
						temperatures[i][0] = ( temperatures_last[i-1][0] + temperatures_last[i  ][1] ) / 2.0;
					else
						temperatures[i][0] = ( temperatures_last[i-1][0] + temperatures_last[i+1][0] + temperatures_last[i  ][1] ) / 3.0;
				}
				temp1 = fmax(fabs(temperatures[i][0] - temperatures_last[i][0]), temp1);
			}
			*/

			#pragma acc loop independent tile(32,32)
			for(int i = 1; i < ROWS - 1; i++)
			{
				// Process all cells between the first and last columns excluded, which each has both left and right neighbours
				for(int j = 1; j < COLUMNS - 1; j++)
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
			
			/*
			#pragma acc loop independent
			for(int i = 1; i <= ROWS_PER_MPI_PROCESS; i++)
			{
				// Process the cell at the last column, which has no right neighbour
				if(temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] != MAX_TEMPERATURE)
				{
					if(i==0)
						temperatures[i][0] = ( 0 + temperatures_last[i+1][0] + temperatures_last[i  ][1] ) / 3.0;
					else if(i==ROWS-1)
						temperatures[i][0] = ( temperatures_last[i-1][0] + 0 + temperatures_last[i  ][1] ) / 3.0;
					else
						temperatures[i][0] = ( temperatures_last[i-1][0] + temperatures_last[i+1][0] + temperatures_last[i  ][1] ) / 3.0;
					
					temp3 = fmax(fabs(temperatures[i][COLUMNS_PER_MPI_PROCESS - 1] - temperatures_last[i][COLUMNS_PER_MPI_PROCESS - 1]), temp3);
				}
			}
			*/
		}

		///////////////////////////////////////////////////////
		// -- SUBTASK 3: CALCULATE MAX TEMPERATURE CHANGE -- //
		///////////////////////////////////////////////////////
		// only need to reduce the values from the 3 subprocesses
		my_temperature_change = fmax(fmax(temp1, temp2), temp3);
		printf("here2\n");
		//////////////////////////////////////////////////////////
		// -- SUBTASK 4: FIND MAX TEMPERATURE CHANGE OVERALL -- //
		//////////////////////////////////////////////////////////
		//MPI_Reduce(&my_temperature_change, &global_temperature_change, 1, MPI_DOUBLE, MPI_MAX, MASTER_PROCESS_RANK, MPI_COMM_WORLD);
		global_temperature_change = my_temperature_change;

		//////////////////////////////////////////////////
		// -- SUBTASK 5: UPDATE LAST ITERATION ARRAY -- //
		//////////////////////////////////////////////////
		#pragma acc kernels loop independent collapse(2)
		for(int i = 0; i <= ROWS; i++)
		{
			for(int j = 0; j < COLUMNS; j++)
			{
				temperatures_last[i][j] = temperatures[i][j];
			}
		}
printf("here3\n");
		///////////////////////////////////
		// -- SUBTASK 6: GET SNAPSHOT -- //
		///////////////////////////////////
		if(iteration_count % SNAPSHOT_INTERVAL == 0)
		{
				printf("Iteration %d: %.18f\n", iteration_count, global_temperature_change);
				#pragma acc update host(temperatures[0:ROWS][0:COLUMNS])
				memcpy(&snapshot[0][0], &temperatures[0][0], ROWS * COLUMNS);
		}
printf("here4\n");
		// Calculate the total time spent processing
		if(my_rank == MASTER_PROCESS_RANK)
		{
			total_time_so_far = MPI_Wtime() - start_time;
		}

		// Send total timer to everybody so they too can exit the loop if more than the allowed runtime has elapsed already
		//MPI_Bcast(&total_time_so_far, 1, MPI_DOUBLE, MASTER_PROCESS_RANK, MPI_COMM_WORLD);

		// Update the iteration number
		iteration_count++;
	}

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
