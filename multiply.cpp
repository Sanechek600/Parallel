#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <string>

#define DISTANCE 10000.0
#define MAX_SIZE 500
#define NUM_FILES_PER_SIZE 10

using namespace std;
using namespace std::chrono;

vector<vector<double>> read_matrix(const string& filename, int rows, int cols) {
    ifstream file(filename);
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            file >> matrix[i][j];
    file.close();
    return matrix;
}

void generate_matrix(const string& filename, int rows, int cols) {
    ofstream file(filename);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-DISTANCE, DISTANCE);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double val = dis(gen);
            file << fixed << setprecision(2) << val << " ";
        }
        file << "\n";
    }

    file.close();
}

vector<vector<double>> multiply_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B, int n, int m, int p) {
    vector<vector<double>> result(n, vector<double>(p, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            for (int k = 0; k < m; ++k)
                result[i][j] += A[i][k] * B[k][j];
    return result;
}

void write_matrix(const string& filename, const vector<vector<double>>& matrix) {
    ofstream file(filename);
    for (const auto& row : matrix) {
        for (double val : row) {
            file << fixed << setprecision(2) << val << " ";
        }
        file << "\n";
    }
    file.close();
}

int main() {
    setlocale(LC_ALL, "");

    ofstream timingFile("timing_results.txt");
    timingFile << "Size\tMean time (sec)\n";

    for (int size = 50; size <= MAX_SIZE; size += 50) {
        int n = size, m = size, p = size;
        double totalTime = 0.0;

        cout << "Processing matrices size: " << size << "x" << size << "...\n";

        for (int i = 1; i <= NUM_FILES_PER_SIZE; ++i) {
            string fileA = "input\\MatrixA(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileB = "input\\MatrixB(" + to_string(size) + ")_" + to_string(i) + ".txt";
            string fileResult = "input\\Result(" + to_string(size) + ")_" + to_string(i) + ".txt";

            generate_matrix(fileA, n, m);
            generate_matrix(fileB, m, p);

            auto A = read_matrix(fileA, n, m);
            auto B = read_matrix(fileB, m, p);

            auto start = high_resolution_clock::now();
            auto result = multiply_matrices(A, B, n, m, p);
            auto end = high_resolution_clock::now();
            totalTime += duration<double>(end - start).count();

            write_matrix(fileResult, result);
        }

        double averageTime = totalTime / NUM_FILES_PER_SIZE;
        timingFile << size << "x" << size << "\t" << fixed << setprecision(6) << averageTime << "\n";
    }

    timingFile.close();
    cout << "\nDone. Execution time stats saved as timing_results.txt\n";

    return 0;
}