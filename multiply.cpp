#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

#define MIN_SIZE 50
#define MAX_SIZE 500
#define SIZE_INCREMENT 50

using namespace std;
using namespace std::chrono;

void generateMatrix(const string& filename, int rows, int cols) {
    ofstream file(filename);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1000000.0, 1000000.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << fixed << setprecision(2) << dis(gen) << " ";
        }
        file << "\n";
    }

    file.close();
}

void writeMatrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (double val : row)
            file << fixed << setprecision(2) << val << " ";
        file << "\n";
    }
    file.close();
}

vector<vector<double>> readMatrix(const string& filename, int rows, int cols) {
    ifstream file(filename);
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i][j];
    file.close();
    return matrix;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    cout << " Process rank: " << rank << " of " << size << endl;

    const int numPairs = 10;

    ofstream timingFile;
    if (rank == 0)
        timingFile.open("timing_results"+to_string(size)+".txt"), timingFile << "Размерность\tСреднее время (сек)\n";

    for (int dim = MIN_SIZE; dim <= MAX_SIZE; dim += SIZE_INCREMENT) {
        int n = dim, m = dim, p = dim;
        double totalTime = 0.0;

        if (rank == 0)
            cout << "Обработка размерности: " << dim << "x" << dim << "...\n";

        for (int pair = 1; pair <= numPairs; ++pair) {
            string fileA = "MatrixA(" + to_string(dim) + ")_" + to_string(pair) + ".txt";
            string fileB = "MatrixB(" + to_string(dim) + ")_" + to_string(pair) + ".txt";
            string fileResult = "Result(" + to_string(dim) + ")_" + to_string(pair) + ".txt";

            vector<vector<double>> A, B;
            vector<double> flatA, flatB(m * p);

            if (rank == 0) {
                generateMatrix(fileA, n, m);
                generateMatrix(fileB, m, p);

                A = readMatrix(fileA, n, m);
                B = readMatrix(fileB, m, p);

                flatA.resize(n * m);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < m; ++j)
                        flatA[i * m + j] = A[i][j];

                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < p; ++j)
                        flatB[i * p + j] = B[i][j];
            }
            else {
                flatB.resize(m * p);
            }

            int rowsPerProc = n / size;
            int extra = n % size;
            int localRows = rowsPerProc + (rank < extra ? 1 : 0);

            vector<int> sendCounts(size), displs(size);
            int offset = 0;
            for (int r = 0; r < size; ++r) {
                int rows = rowsPerProc + (r < extra ? 1 : 0);
                sendCounts[r] = rows * m;
                displs[r] = offset;
                offset += sendCounts[r];
            }

            vector<double> localA(localRows * m);
            vector<double> localC(localRows * p);

            MPI_Bcast(flatB.data(), m * p, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatterv(flatA.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                localA.data(), localRows * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            auto start = high_resolution_clock::now();

            for (int i = 0; i < localRows; ++i) {
                for (int j = 0; j < p; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < m; ++k)
                        sum += localA[i * m + k] * flatB[k * p + j];
                    localC[i * p + j] = sum;
                }
            }

            vector<int> recvCounts(size), recvDispls(size);
            offset = 0;
            for (int r = 0; r < size; ++r) {
                int rows = rowsPerProc + (r < extra ? 1 : 0);
                recvCounts[r] = rows * p;
                recvDispls[r] = offset;
                offset += recvCounts[r];
            }

            vector<double> flatC;
            if (rank == 0) flatC.resize(n * p);

            MPI_Gatherv(localC.data(), localRows * p, MPI_DOUBLE,
                flatC.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            if (rank == 0) {
                auto end = high_resolution_clock::now();
                totalTime += duration<double>(end - start).count();

                vector<vector<double>> result(n, vector<double>(p));
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < p; ++j)
                        result[i][j] = flatC[i * p + j];

                writeMatrix(fileResult, result);
            }
        }

        if (rank == 0) {
            double avgTime = totalTime / numPairs;
            timingFile << dim << "x" << dim << "\t" << fixed << setprecision(6) << avgTime << "\n";
        }
    }

    if (rank == 0)
        timingFile.close();

    MPI_Finalize();
    return 0;
}