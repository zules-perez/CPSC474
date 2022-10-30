/*
* multiplies matrices A and B and inserts the product in matrix C using MPI.
* Assumes matrices A and B are dense. Will only use square N x N matrices.
* The size of the matrices must be divisible by the number of processors or
* matrix multiplication will stop.
*/
#include <mpi.h>
#include <iostream>
#include <iomanip>

#define N 20

// Function to print out a matrix.
void print(int matrix[N][N]) {
    for (int x = 0; x < N; x++) {
        std::cout << "\n\t\t";
        for (int y = 0; y < N; y++) {
            std::cout << std::setw(3);
            std::cout << matrix[x][y] << " ";
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    int Matrix_A[N][N];

    for (auto &x: Matrix_A) {
        for (int & y : x) {
            y = rand() % 10;
        }
    }

    int Matrix_B[N][N];

    for (auto & x : Matrix_B) {
        for (int & y : x) {
            y = rand() % 10;
        }
    }

    int Matrix_C[N][N];

// Initialize MPI
    MPI_Init(&argc, &argv);
// Determine number of processes.
    MPI_Comm_size(MPI_COMM_WORLD, &size);
// Determine process identifier.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "\n\tProcess " << rank << " out of " << size << std::endl;
    if (N % size != 0) {
        if (rank == 0)
            std::cout << "Cannot divide size of matrix by the number of processors" << std::endl;
        MPI_Finalize();
        return 0;
    }

// MPI_Bcast(&sendbuf, count, datatype, root, comm)
// Matrix B is copied to every processor.
    MPI_Bcast(Matrix_B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

// MPI_Scatter(void* send_data, int send_count,MPI_Datatype send_datatype,
// void* recv_data, int recv_count,MPI_Datatype recv_datatype, int root,
// MPI_Comm communicator)

// Replaced A[rank*(N/size)] with MPI_IN_PLACE in MPI_SCATTER recvbuff
// Matrix A is divided into blocks of rows and distributed among the processors.
    MPI_Scatter(Matrix_A, (N * N) / size, MPI_INT, MPI_IN_PLACE, (N * N) / size, MPI_INT, 0, MPI_COMM_WORLD);

// MPI_BARRIER(MPI_Comm communicator)
// Synchronize all processes at barrier.
    MPI_Barrier(MPI_COMM_WORLD);

// Matrix*Matrix Multiplication
    for (int x = (rank * N) / size; x < (rank + 1) * N / size; x++) {
        for (int y = 0; y < N; y++) {
            Matrix_C[x][y] = 0;
            for (int z = 0; z < N; z++) {
                Matrix_C[x][y] += Matrix_A[x][z] * Matrix_B[z][y];
            }
        }
    }
// MPI_Gather(void* send_data, int send_count, MPI_Datatype send_datatype,
// void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root,
// MPI_Comm communicator)

// Replaced C[rank*(N/size)] with MPI_IN_PLACE in MPI_GATHER sendbuf
// Matrix C gathers the distributed blocks of rows from the processors.

    MPI_Gather(MPI_IN_PLACE, (N * N) / size, MPI_INT, Matrix_C, (N * N) / size, MPI_INT, 0,
               MPI_COMM_WORLD);
// Synchronize all processes at barrier.
    MPI_Barrier(MPI_COMM_WORLD);
// Checks if it is at the root process (rank == 0) and
// if prints out the matrices A*B = C. (if rank == 0)
    if (rank == 0) {
        std::cout << "\n\n\t" << " A * B = C \n" << std::endl;
        std::cout << std::endl;
        std::cout << "\t A = ";
        print(Matrix_A);
        std::cout << std::endl << std::endl;
        std::cout << "\t B = ";
        print(Matrix_B);
        std::cout << std::endl << std::endl;
        std::cout << "\t C = ";
        print(Matrix_C);
        std::cout << std::endl << std::endl;
    }
// Need this to shut down MPI.
    MPI_Finalize();
    return 0;
}