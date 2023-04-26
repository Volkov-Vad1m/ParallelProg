#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    long N = atoi(argv[1]); // matrix size
    
    int rank, size, i, j;
    int matrix[N][N], transposed[N][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank of current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); 


    if (rank == 0) {
        printf("Matrix:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                matrix[i][j] = i * N + j;
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    // Distribute matrix A to all processes
    MPI_Bcast(&matrix[0][0], N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Transpose matrix A
    int local[N/size][N];
    for (i = 0; i < N/size; i++) {
        for (j = 0; j < N; j++) {
            local[i][j] = matrix[j][i + rank*N/size];
        }
    }

    // Gather transposed matrix B
    MPI_Gather(&local[0][0], N*N/size, MPI_INT, &transposed[0][0], N*N/size, MPI_INT, 0, MPI_COMM_WORLD);

    // Print transposed matrix B on root process
    if (rank == 0) {
        printf("Result:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", transposed[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();


    return 0;
}