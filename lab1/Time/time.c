#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MESSAGE_SIZE 1000000  // Size of the message in bytes

int main(int argc, char** argv) {
    int rank, size;
    char* message;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        fprintf(stderr, "[err] 2 MPI processes are required\n");
        MPI_Finalize();
        exit(1);
    }
    
    message = (char*) malloc(MESSAGE_SIZE);
    if (rank == 0) {
        printf("[proc][0] Sending message of size %d bytes from process 0 to process 1\n", MESSAGE_SIZE);
        start_time = MPI_Wtime();
        MPI_Send(message, MESSAGE_SIZE, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        end_time = MPI_Wtime();
        printf("[proc][0] Message sent in %f seconds\n", end_time - start_time);
    } else if (rank == 1) {
        MPI_Recv(message, MESSAGE_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[proc][1] Received message of size %d bytes on process 1\n", MESSAGE_SIZE);
    }
    
    free(message);
    MPI_Finalize();
    return 0;
}