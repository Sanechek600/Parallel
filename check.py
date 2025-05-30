import numpy as np
import matplotlib.pyplot as plt
import os
import re

INPUT_DIR = ""
LOG_FILE = "check.log"
NUM_FILES_PER_SIZE = 10
MAX_SIZE = 500
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
    sizes = []
    times = []
    with open(filename, 'r', encoding='cp1251') as f:
        lines = f.readlines()[1:]
        for line in lines:
            match = re.match(r"(\d+x\d+)\s+([0-9.]+)", line.strip())
            if match:
                size, time = match.groups()
                sizes.append(size)
                times.append(float(time))
    return sizes, times


def plot_multiple_timings(files_labels_colors, out_filename):
    plt.figure(figsize=(10, 6))
    
    for filename, label, color in files_labels_colors:
        if not os.path.exists(filename):
            print(f"[Warning] {filename} not found and will be skipped.")
            continue
        sizes, times = parse_timing_results(filename)
        numeric_sizes = [int(s.split('x')[0]) for s in sizes]
        plt.plot(numeric_sizes, times, marker='o', label=label, color=color)

    plt.title("Multiplication time comparison")
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Mean time (sec)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(out_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved as: {output_path}")

    plt.show()

def main():
    sizes = [i*SIZE_INCREMENT for i in range(1, MAX_SIZE//SIZE_INCREMENT + 1)]

    print("*** Checking multiplication results ***")
    log_path = os.path.join(LOG_FILE)
    with open(log_path, "w", encoding="utf-8") as log_file:
        for size in sizes:
            verify_multiplication(size, log_file)

    print(f"\nCkeck results saved as: {log_path}")

    print("\n*** Creating time graph ***")
    files_labels_colors = [
        ("timing_results.txt", "Base algorithm", "blue"),
        ("timing_results4.txt", "4 processes", "green"),
        ("timing_results12.txt", "12 processes", "red")
    ]
    plot_multiple_timings(files_labels_colors, "timing_plot_comparison.png")

    files_labels_colors = [
        ("1K.txt", "Base algorithm", "blue"),
        ("4K.txt", "4 processes", "green"),
        ("12K.txt", "12 processes", "red")
    ]
    plot_multiple_timings(files_labels_colors, "korolev_comparison.png")


if __name__ == "__main__":
    main()