#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>

#define LR_MIN 50
#define LR_MAX 500
#define HR_MIN 1000
#define HR_MAX 4000
#define LR_INC 50
#define HR_INC 500

using namespace std;

using Matrix = vector<vector<float>>;

Matrix generateMatrix(int rows, int cols) {
    Matrix mat(rows, vector<float>(cols));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i][j] = dis(gen);
    return mat;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes;
    for (int dim = LR_MIN; dim <= LR_MAX; dim += LR_INC)
        sizes.push_back(dim);
    for (int dim = HR_MIN; dim <= HR_MAX; dim += HR_INC)
        sizes.push_back(dim);

    for (int dim : sizes) {
        vector<double> timings;
        for (int run = 0; run < 5; ++run) {
            Matrix A, B;
            if (rank == 0) {
                A = generateMatrix(dim, dim);
                B = generateMatrix(dim, dim);
            }

            Matrix B_shared(dim, vector<float>(dim));
            for (int i = 0; i < dim; ++i)
                MPI_Bcast(rank == 0 ? B[i].data() : B_shared[i].data(), dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
            if (rank != 0) B = std::move(B_shared);

            int baseRows = dim / size;
            int extra = dim % size;
            int myRows = baseRows + (rank < extra ? 1 : 0);

            vector<int> sendCounts(size), displs(size);
            int offset = 0;
            for (int i = 0; i < size; ++i) {
                sendCounts[i] = baseRows + (i < extra ? 1 : 0);
                displs[i] = offset;
                offset += sendCounts[i];
            }

            Matrix localA(myRows, vector<float>(dim));

            if (rank == 0) {
                for (int i = 1; i < size; ++i) {
                    for (int r = 0; r < sendCounts[i]; ++r) {
                        MPI_Send(A[displs[i] + r].data(), dim, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                    }
                }
                for (int r = 0; r < myRows; ++r)
                    localA[r] = A[r];
            }
            else {
                for (int r = 0; r < myRows; ++r) {
                    MPI_Recv(localA[r].data(), dim, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = chrono::high_resolution_clock::now();

            Matrix localC(myRows, vector<float>(dim, 0.0f));
            for (int i = 0; i < myRows; ++i) {
                for (int j = 0; j < dim; ++j) {
                    for (int k = 0; k < dim; ++k)
                        localC[i][j] += localA[i][k] * B[k][j];
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = chrono::high_resolution_clock::now();

            if (rank == 0) {
                chrono::duration<double> duration = end - start;
                timings.push_back(duration.count());
                cout << "Size: " << dim << "x" << dim << ", launch " << run + 1
                    << ", time: " << duration.count() << " s" << endl;
            }
        }

        if (rank == 0) {
            double avg = accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
            cout << "Mean for size " << dim << "x" << dim << ": "
                << avg << " s" << endl << endl;
        }
    }

    MPI_Finalize();
    return 0;
}