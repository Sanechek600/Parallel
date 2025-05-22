import numpy as np
import matplotlib.pyplot as plt
import os
import re

INPUT_DIR = "input"
LOG_FILE = "check.log"
NUM_FILES_PER_SIZE = 10
MAX_SIZE = 300
SIZE_INCREMENT = 50

def read_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        matrix = [list(map(float, line.strip().split())) for line in lines]
    return np.array(matrix)


def verify_multiplication(size, log_file=None):
    all_correct = True

    for i in range(1, NUM_FILES_PER_SIZE + 1):
        filename_A = os.path.join(INPUT_DIR, f"MatrixA({size})_{i}.txt")
        filename_B = os.path.join(INPUT_DIR, f"MatrixB({size})_{i}.txt")
        filename_Result = os.path.join(INPUT_DIR, f"Result({size})_{i}.txt")

        if not all(os.path.exists(f) for f in [filename_A, filename_B, filename_Result]):
            message = f"[MISSING] MatrixA/MatrixB/Result for size {size}, set {i}."
            print(message)
            if log_file:
                log_file.write(message + "\n")
            all_correct = False
            continue

        A = read_matrix(filename_A)
        B = read_matrix(filename_B)
        Result = read_matrix(filename_Result)
        calculated = np.matmul(A, B)

        if np.allclose(calculated, Result, atol=1e-6):
            message = f"[OK] Multiplication of {size}, set {i} is correct."
        else:
            message = f"[Error] Multiplication of {size}, set {i} is incorrect."
            all_correct = False

        print(message)
        if log_file:
            log_file.write(message + "\n")

    return all_correct


def parse_timing_results(filename):
    times = []
    with open(filename, 'r', encoding='cp1251') as f:
        lines = f.readlines()[1:]
        for line in lines:
            match = re.match(r"(\d+x\d+)\s+([0-9.]+)", line.strip())
            if match:
                time = match.groups()[1]
                times.append(float(time))
    return times


def plot_timing(sizes, times):
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', color='blue')
    plt.title("Operation time to matrix size relation")
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Mean time (sec)")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join("timing_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved as: {output_path}")

    plt.show()


def main():
    timing_file = "timing_results.txt"

    if not os.path.exists(timing_file):
        print(f"{timing_file}: file not found.")
        return

    times = parse_timing_results(timing_file)
    sizes = [i*SIZE_INCREMENT for i in range(1, MAX_SIZE//SIZE_INCREMENT + 1)]

    print("*** Checking multiplication results ***")
    log_path = os.path.join(LOG_FILE)
    with open(log_path, "w", encoding="utf-8") as log_file:
        for size in sizes:
            verify_multiplication(size, log_file)

    print(f"\nCkeck results saved as: {log_path}")

    print("\n*** Creating time graph ***")
    plot_timing(sizes, times)


if __name__ == "__main__":
    main()