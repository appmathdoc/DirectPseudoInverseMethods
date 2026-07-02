import numpy as np
import scipy.linalg
import time

def generate_rank_r_matrix(m, n):
    """
    Generates an m x n random matrix with the specified rank constraints.
    """
    if m == 10 or n == 10:
        r = 10
    else:
        r = min(m, n) // 2
        
    # Generating rank r matrix by multiplying (m x r) and (r x n) matrices
    # Gaussian random matrices are almost surely full rank.
    L = np.random.randn(m, r)
    R = np.random.randn(r, n)
    return L @ R

def run_benchmarks(fast_pinv, n_trials=100, skip_10k = False):
    if(skip_10k):
        m_values = [10, 100, 1000]
        n_values = [10, 100, 1000]
    else:
        m_values = [10, 100, 1000, 10000]
        n_values = [10, 100, 1000, 10000]

    
    # Dictionary to store results for table generation
    # format: results[(m, n)] = (mean_speedup, std_speedup)
    results = {}

    print(f"--- Starting Benchmark: {n_trials} Trials per (m, n) Pair ---")
    print("Warning: 10000x10000 matrix testing may take significant time.\n")

    for m in m_values:
        for n in n_values:
            print(f"Testing Dimensions: m={m:<5} x n={n:<5} ", end="", flush=True)
            
            speedups = np.zeros(n_trials)
            
            for i in range(n_trials):
                # 1. Generate Matrix
                A_test = generate_rank_r_matrix(m, n)
                
                # 2. Benchmark Fast4StepPinv
                start_fast = time.time()
                _ = fast_pinv(A_test)
                time_fast = time.time() - start_fast
                
                # 3. Benchmark SciPy
                start_scipy = time.time()
                _ = scipy.linalg.pinv(A_test)
                time_scipy = time.time() - start_scipy
                
                # 4. Calculate Speedup (Scipy Time / Fast Time)
                # Adding a tiny epsilon to prevent division by zero on very fast tiny matrices
                speedups[i] = time_scipy / (time_fast + 1e-12)

            # Calculate Statistics
            mean_su = np.mean(speedups)
            std_su = np.std(speedups)
            
            results[(m, n)] = (mean_su, std_su)
            print(f"-> Speedup: {mean_su:.2f}x ± {std_su:.2f}")

    return m_values, n_values, results

def print_speedup_table(m_values, n_values, results):
    """
    Prints a formatted grid of the Speedup Means and Standard Deviations.
    """
    print("\n" + "="*80)
    print("   PSEUDOINVERSE SPEEDUP BENCHMARK (FastPinv vs SciPy SVD)")
    print("   Values > 1.0 indicate FastPinv is faster.")
    print("="*80)
    
    # Print Header Row
    header = f"{'m \\ n':<12}|"
    for n in n_values:
        header += f" {n:<14} |"
    print(header)
    print("-" * len(header))
    
    # Print Data Rows
    for m in m_values:
        row_str = f"{m:<12}|"
        for n in n_values:
            mean_su, std_su = results[(m, n)]
            cell = f"{mean_su:5.2f}x ± {std_su:4.2f}"
            row_str += f" {cell:<14} |"
        print(row_str)
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # You can change n_trials here if the 10k x 10k iterations take too long
    print("Using FastPinvA from Method1ScipyPinvB.py")
    from Method1ScipyPinvB import *
    m_vals, n_vals, test_results = run_benchmarks(FastPinvA, n_trials=100, skip_10k = True)
    print_speedup_table(m_vals, n_vals, test_results)