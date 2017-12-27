#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>
#include <math.h>

#define ROWS 16
#define COLS 16
#define TEMP 50
#define I_FIX 7
#define J_FIX 7
#define EPS 1e-3


void computeFreshValues(double **oldMat, double ** newMat, double nBot[], double nTop[], int rows, int cols, int rank, int master, int lastrank, int source){
	if(rank != master){
		for(int i = 1; i < COLS-1; i++){
			newMat[0][i] = 0.25*(oldMat[0][i-1] + oldMat[0][i+1] + nTop[i] + oldMat[1][i+1]);
		}
	}
	
	for(int i = 1; i < rows-1; i++){
		for(int j = 1; j < cols-1; j++){
			newMat[i][j] = 0.25*(oldMat[i][j-1] + oldMat[i][j+1] + oldMat[i-1][j] + oldMat[i+1][j]);
		}
	}
	if(rank != lastrank){
		for(int i = 1; i < COLS-1; i++){
			newMat[rows-1][i] = 0.25*(oldMat[rows-1][i-1] + oldMat[rows-1][i+1] + oldMat[rows-2][i] + nBot[i]);
		}	
	}
	if(rank == source && rank != lastrank){
		int offset = I_FIX%rows;
		newMat[offset][J_FIX] = TEMP;
	}

	if(rank == source && rank == lastrank){
		int temp = ROWS/(lastrank+1);
		int offset = I_FIX - temp*(lastrank);
		newMat[offset][J_FIX] = TEMP;
	}
	
}


void copy_matrix(double** old, double** newMat, int rows, int cols){
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			old[i][j] = newMat[i][j];
		}
	}
}

int getSourceRank(int chunk){
	int source = (I_FIX+1)/chunk;
	if(((I_FIX+1)%chunk) !=0 ) source++;
	return source-1;
}

double** alloc_matrix(int rows, int cols){
	double** matrix;
	matrix = (double**) malloc(rows * sizeof(double *));
	matrix[0] = (double*) calloc(rows * cols, sizeof(double));
	for (int i = 1; i < rows; i++)
		matrix[i] = matrix[0] + i*cols;
	return matrix;
}

double max_abs(double** old, double** newMat, int rows, int cols){
	double max_val = DBL_MIN;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			if(fabs(old[i][j] - newMat[i][j]) > max_val){
				max_val = fabs(old[i][j] - newMat[i][j]);
			}
		}
	}
	return max_val;
}



int main(int argc, char *argv[]){

	// SECTION 1 : Initialize variables for all processes
	int size,rank,master = 0,lastrank, perProcessSize, rows_in_last_process,tag = 0;
	int sourceRank, reached_threshold;
	double globalMax;



	// SECTION 2 : MPI Begins
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;


	// SECTION 2.1 : Setup division of matrices
	lastrank = size-1;
	perProcessSize = ROWS/size;
	if(rank == lastrank)
		perProcessSize+=ROWS%size;


	// SECTION 2.2 : Initialize local copies of the matrices
	double** localMatrix = alloc_matrix(perProcessSize, COLS);;
	double** newLocalMatrix = alloc_matrix(perProcessSize, COLS);;
	
	// SECTION 2.3 : Setup allocation process for all processes
	double** oldMat;
	sourceRank = getSourceRank(perProcessSize);

	if(sourceRank == size)
		sourceRank = lastrank;
	if(rank == master){

		oldMat = alloc_matrix(ROWS, COLS);
		oldMat[I_FIX][J_FIX] = TEMP;
		int index = perProcessSize;

		for(int p = 1; p < size-1; p++){
			for(; index < (p+1)*perProcessSize;index++){
				MPI_Send(oldMat[index], COLS, MPI_DOUBLE, p, tag, MPI_COMM_WORLD);
			}
		}

		for(; index < ROWS; index++){
			MPI_Send(oldMat[index], COLS, MPI_DOUBLE, lastrank, tag, MPI_COMM_WORLD);
		}

		for(int i = 0; i < perProcessSize; i++){
			for(int j = 0; j < COLS; j++){
				localMatrix[i][j] = oldMat[i][j];
			}
		}

	}

	if(rank != master){
		for(int j = 0; j < perProcessSize; j++){
			MPI_Recv(localMatrix[j], COLS, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);


	// SECTION 3 : Initiliaze local Ghost cells
	double nTop[COLS] = {0};
	double nBot[COLS] = {0};


	while(1){
		if(rank%2 == 0 && rank != master){
			MPI_Recv(nTop, COLS, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
		}
		if(rank%2!=0 && rank != lastrank){
			MPI_Send(localMatrix[perProcessSize-1], COLS, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank%2 == 0 && rank != lastrank){
			MPI_Send(localMatrix[perProcessSize-1], COLS, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD);
		}
		if(rank%2!=0){
			MPI_Recv(nTop, COLS, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &status);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank%2==0 && rank != lastrank){
			MPI_Recv(nBot, COLS, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		}
		if(rank!=0){
			MPI_Send(localMatrix[0], COLS, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		if(rank%2 == 0 && rank != master){
			MPI_Send(localMatrix[0], COLS, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD);
		}
		if(rank%2!=0 && rank != lastrank){
			MPI_Recv(nBot, COLS, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &status);
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		
		
		computeFreshValues(localMatrix, newLocalMatrix, nBot, nTop, perProcessSize, COLS, rank, master, lastrank, sourceRank);
		

		MPI_Barrier(MPI_COMM_WORLD);

		double max_diff = max_abs(localMatrix, newLocalMatrix, perProcessSize, COLS);

		MPI_Reduce(&max_diff, &globalMax, 1, MPI_DOUBLE, MPI_MAX, master, MPI_COMM_WORLD);
		
		if(rank == master){
			if(globalMax < EPS){
				reached_threshold = 1;
			}
			else{
				reached_threshold = 0;
			}
		}
		
		
		MPI_Bcast(&reached_threshold, 1, MPI_INT, master, MPI_COMM_WORLD);
		
		copy_matrix(localMatrix, newLocalMatrix, perProcessSize, COLS);
		
		if(reached_threshold) break;
		
		
	}
	
	if(rank == master){
		int index = perProcessSize;
		for(int p = 1; p < size-1; p++){
			for(; index < (p+1)*perProcessSize;index++){
				MPI_Recv(oldMat[index], COLS, MPI_DOUBLE, p, tag, MPI_COMM_WORLD, &status);
			}
		}
		for(; index < ROWS; index++){
			MPI_Recv(oldMat[index], COLS, MPI_DOUBLE, lastrank, tag, MPI_COMM_WORLD, &status);
		}
		for(int i = 0; i < perProcessSize; i++){
			for(int j = 0; j < COLS; j++){
				oldMat[i][j] = localMatrix[i][j];
			}
		}
	}
	else{
		for(int j = 0; j < perProcessSize; j++){
			MPI_Send(localMatrix[j], COLS, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
		}
	}
	
	if(rank == master){
		FILE *fp;
		fp = fopen("output.txt", "w+");

		
		for(int i = 0; i < ROWS; i++){
			for(int j = 0; j <COLS; j++){
				//printf("%f ", oldMat[i][j]);
				fprintf(fp, "%f\t", oldMat[i][j]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	
	free(localMatrix[0]);
	free(localMatrix);

	free(newLocalMatrix[0]);
	free(newLocalMatrix);

	if(rank == master){
		free(oldMat[0]);
		free(oldMat);
	}
	MPI_Finalize();
	return 0;
}
