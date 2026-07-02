import numpy as np
from scipy.linalg import pinv, norm
from Method3ExplicitBplusViaCython import *
import sys
import time

import os

def clear_console():
    # Use 'cls' for Windows, 'clear' for Mac and Linux
    os.system('cls' if os.name == 'nt' else 'clear')
    # Use clear_output if inside jupyter shell
    try:
        from IPython.display import clear_output
        clear_output()
    except:
        pass

def TestPinv(A, artificial_tol = 18.4, test_lbl = "", dest = sys.stdout, 
                max_rows = 10,  max_cols = 20 ):

    m = min(A.shape[0], max_rows)
    n = min(A.shape[1], max_cols)

    if dest == sys.stdout:
        clear_console()
    print(f"\n\nTEST {test_lbl}:", file = dest )
    print(f"Matrix A: {A.shape[0]} x {A.shape[1]}: (First {m} rows and {n} columns shown)", file = dest )
    print(np.round(A, 3)[:m, :n], file = dest )


    ## Time for standard scipy.linalg.pinv 
    start_time = time.time()
    A_pinv_svd = pinv(A)  ## scipy
    np_time = time.time() - start_time

    ## Time for our Method
    start_time = time.time()
    A_pinv_fast = FastPinv(A)
    fast_time = time.time() - start_time

    print(f"\nScipy linalg.pinv Execution Time: {np_time:.4f} seconds")
    print(f"Lanczos  FastPinv Execution Time: {fast_time:.4f} seconds")

    ## Print Result
    print("\nMatrix A_pinv_fast:", file = dest )
    print(np.round(A_pinv_fast[:m,:n], 3), file = dest )
        
    print("\nResidual Error (Fast vs SVD):", file = dest )
    error = np.linalg.norm(A_pinv_fast - A_pinv_svd)
    print(f"{error:.4e}", file = dest )
    
    # 5. Verify Penrose Conditions
    print("\nVerifying Penrose Conditions for our Fast Solver:", file = dest )
    print("1. A X A = A      -> Error:", np.linalg.norm(A @ A_pinv_fast @ A - A), file = dest )
    print("2. X A X = X      -> Error:", np.linalg.norm(A_pinv_fast @ A @ A_pinv_fast - A_pinv_fast), file = dest )
    
    AX = A @ A_pinv_fast
    print("3. (AX)* = AX     -> Error:", np.linalg.norm(AX.T - AX), file = dest )
    
    XA = A_pinv_fast @ A
    print("4. (XA)* = XA     -> Error:", np.linalg.norm(XA.T - XA), file = dest )

    return A_pinv_fast

def run_random_tests(n_tests=1000, mn_bounds = (1_000, 100), rank_bound = None, 
                     tol=1e-7, log10_cond = 5, stop_on_fail = False):
    """
    Stress-tests the pinv_bidiagonal implementation against scipy.linalg.pinv
    using randomized sparse bidiagonal matrices.
    """
    print(f"Running {n_tests} random tests: '.' = good, ':' = cond 1, '|' = fail")
    errs = dict()

    ms = []
    ns = []
    rs = [] 
    
    i = 0
    while i < n_tests: 
        # 1. Randomly sample dimensions between 2 and 1000
        m = np.random.randint(2, mn_bounds[0])
        n = np.random.randint(2, mn_bounds[1])
        min_mn = min(m,n)
        
        if(rank_bound): 
            rank_bound = min(rank_bound,min_mn)
        else:
            rank_bound = min_mn
        r = np.random.randint(min(5,rank_bound//2),rank_bound)
        ms.append(m)
        ns.append(n)
        rs.append(r)
        
        A = np.random.randn(m, r) @ np.random.randn(r, n) 
        sci_pinv = pinv(A)
            
        ## continue if conditioning an issue
        if( np.log10( norm(A)*norm(sci_pinv) ) > log10_cond ):
            continue
        else:
            i+=1

        # 5. Compute both pseudoinverses
        try:
            my_pinv = FastPinv( A)
        except Exception as e:
            print(f"\nCrash at iteration {i} (Shape: {m}x{n}): {e}")
            print("Matrix A:")
            print(A)
            break
            
        # 6. Check tolerance
        diff = np.linalg.norm(sci_pinv - my_pinv)
        
        if diff < tol:
            print(".", end="")
            sys.stdout.flush()
        else:
            cond1_test = np.linalg.norm( A @ my_pinv @ A - A )
            errs[i] = (A, r, diff, cond1_test)
            if( cond1_test < tol ):
                print(":", end = "")
            else:
                print("|", end = "")
            if( stop_on_fail):        
                print(f"\nTolerance exceeded at iteration {i}!")
                print(f"Shape: {m}x{n} | Diff norm: {diff:.4e}")
                print("Failed Matrix A:")
                print(A)
                return A
            
        # 7. Print progress every 100 iterations
        if i % 100 == 0:
            print(f" {i} iterations thus far")
            
    else:
        ms = np.array(ms)
        ns = np.array(ns)
        rs = np.array(rs)
        hw = ms/ns
        
        print(f"\nAll {n_tests} tests completed. ")
        print(f"\nm range: {ms.min()} to {ms.max()}")
        print(f"n range: {ns.min()} to {ns.max()}")
        print(f"Height to Width Ratio Range:  {hw.min():.4f} to {hw.max():.4f}")
        print(f"\nRank range: {rs.min()} to {rs.max()}")
        print("\n\nCheck errs for information: errs[key] = (A,r,diff,cond1_test)")
    return errs
    
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    n_diag = 7
    B_shape = (7,9)
    
    B2 = np.diag([2**(2-i) for i in range(B_shape[0])], k=1)
   
    B3 = np.zeros(B_shape, dtype = float) 
    B3 = np.zeros((n_diag+3, n_diag+1))
    B3[:n_diag+1,:n_diag+1] = B2#[:-1]

    np.random.seed(42)
    m, n = 50, 40
    
    # Generate an arbitrary, dense random matrix
    A_dense = np.random.randn(m, n) * 10.0
    lbl_dense = f"Testing End-to-End Fast Pseudoinverse on a {m}x{n} Dense Matrix..."

    tests = { 'Testing pathological matrix...':
        np.array([
            [1.,  1.5, 0.,  0.,  0.,  0.,  0. ],
            [0.,  2.,  2.5, 0.,  0.,  0.,  0. ],
            [0.,  0.,  0.,  0.,  0.,  0.,  0. ], # The Problem Row!
            [0.,  0.,  0.,  4.,  4.5, 0.,  0. ],
            [0.,  0.,  0.,  0.,  5.,  0.,  0. ],
            [0.,  0.,  0.,  0.,  0.,  6.,  6.5]
        ]),
        "1":np.array([ [8., 1., 0., 0., 0., 0., 0., 0.],
                   [0., 8., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 8., 5., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 5., 0., 0., 0.],
                   [0., 0., 0., 0., 1., 5., 0., 0.],
                   [0., 0., 0., 0., 0., 4., 8., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]]), 
        "2":B2, "3":B3,
        "4":np.array([ [7., 0., 0., 0., 0.],
               [0., 0., 8., 0., 0.],
               [0., 0., 0., 6., 0.],
               [0., 0., 0., 5., 0.],
               [0., 0., 0., 0., 1.]]),
        "5":np.array([ [9., 2., 0., 0., 0., 0.],
               [0., 0., 4., 0., 0., 0.],
               [0., 0., 0., 5., 0., 0.],
               [0., 0., 0., 5., 7., 0.],
               [0., 0., 0., 0., 2., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]]),
        "6":np.array([ [8., 1., 0., 0., 0., 0., 0., 0.],
               [0., 8., 1., 0., 0., 0., 0., 0.],
               [0., 0., 8., 5., 0., 0., 0., 0.],
               [0., 0., 0., 0., 5., 0., 0., 0.],
               [0., 0., 0., 0., 1., 5., 0., 0.],
               [0., 0., 0., 0., 0., 4., 8., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0.]]),
        "7-Small, Ill-conditioned Matrix":np.array([  [ 41.9, -55.8,  69.8],
                [ 59.7,  76.1,  21.7],
                [ 15.5,   1.6,  14.7]]),
        lbl_dense:A_dense  }
    
    for lbl, arr in tests.items():
        Apinv_fast = TestPinv(arr, test_lbl = lbl)    
        cont = input("Enter q to quit. Any other key to continue: ")
        if( cont.lower() in ['q','Q']):
            raise KeyboardInterrupt("Testing Interrupted")

    clear_console()

    print("Random low rank matrices testing")
    ## Random Testing 
    # You can call this directly. 
    # (Default tolerance set to 1e-7 due to floating point accumulation in 1000x1000 matrices)
    errs = run_random_tests(n_tests=1000, tol=1e-7, rank_bound = 50)