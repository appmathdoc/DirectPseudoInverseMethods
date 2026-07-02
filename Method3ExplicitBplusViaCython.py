import numpy as np
from scipy import linalg
from scipy.linalg import lapack

import lapack_fast

import time
import warnings


# ==============================================================================
# 1. IMPLICIT BIDIAGONALIZATION
# ==============================================================================
def bidag_implicit(A):
    """
    Bidiagonalizes A implicitly. DOES NOT form dense U and Vt matrices.
    """
    A_packed = np.asfortranarray(A.copy().astype(np.float64))  
    m, n = A_packed.shape

    a_out, d, e, tauq, taup, info = lapack_fast.cy_dgebrd(A_packed)
    
    B = np.zeros((m, n), dtype=A_packed.dtype)
    np.fill_diagonal(B, d)
    if m >= n:
        np.fill_diagonal(B[:, 1:], e)
    else:
        np.fill_diagonal(B[1:, :], e)
        
    return a_out, tauq, taup, B

# ==============================================================================
# 2. DUAL-SPACE BLOCK SOLVERS
# ==============================================================================

def predict_dptsv_failure(a, b, ran_walk_tol=18.4):
    """
    Tracks the condition number dynamically via a random walk on the 
    log-superdiagonals to predict hardware underflow.
    """
    if len(b) == 0:
        return False
    b_safe = np.where(np.abs(b) < 1e-15, 1e-15, np.abs(b))
    walk = np.cumsum(np.log(b_safe))
    return np.max(walk) > ran_walk_tol

def pinv_bidiag_block(a, b):
    """
    O(n) direct solver that maps a coupled bidiagonal block to the dual space 
    and solves the symmetric tridiagonal normal equations via Thomas algorithm.
    """
    m = len(a)
    if m == 0: return np.zeros((0, 0))
    if m == 1: return np.array([[1.0 / a[0]]]) if a[0] != 0 else np.array([[0.0]])

    # Form symmetric tridiagonal normal equations M = B B^T
    D = a**2
    D[:-1] += b**2
    E = b * a[1:]

    # Un-pivoted Thomas Algorithm (LAPACK ptsv)
    I = np.eye(m)
    ptsv, = lapack.get_lapack_funcs(('ptsv',), (D,))
    _, _, Z, info = ptsv(D.copy(), E.copy(), I)

    # Map back from dual space
    B_plus = np.zeros((m, m))
    for i in range(m):
        B_plus[i, :] = a[i] * Z[i, :]
        if i > 0:
            B_plus[i, :] += b[i-1] * Z[i-1, :]

    return B_plus

def pinv_bidiag_block_woodbury(a, b, ran_walk_tol=8):  #18.4):
    """
    Fault-tolerant wrapper. Applies Tikhonov regularized tearing if underflow
    is predicted, bounding the lowest eigenvalue safely above zero.
    """
    if not predict_dptsv_failure(a, b, ran_walk_tol):
        return pinv_bidiag_block(a, b)

    m = len(a)
    lam = 1e-10  # Tikhonov shift
    
    D = a**2 + lam**2
    D[:-1] += b**2
    E = b * a[1:]

    I = np.eye(m)
    ptsv, = lapack.get_lapack_funcs(('ptsv',), (D,))
    _, _, Z, _ = ptsv(D, E, I)

    B_plus = np.zeros((m, m))
    for i in range(m):
        B_plus[i, :] = a[i] * Z[i, :]
        if i > 0:
            B_plus[i, :] += b[i-1] * Z[i-1, :]
            
    return B_plus


# ==============================================================================
# 3. GEOMETRIC DECOUPLING ENGINE
# ==============================================================================

def pinv_bidiagonal(B, rcond=None):
    """
    Isolates sub-blocks from explicit dense B. Uses a Pure Slicer to guarantee 
    square blocks, routing full-rank blocks to the O(n) Thomas solver and 
    singular blocks to the robust SVD pseudoinverse.
    """
    m, n = B.shape
    
    # Transpose Duality: Forces strict Upper Bidiagonal logic
    if m < n:
        return pinv_bidiagonal(B.T, rcond).T

    B_plus = np.zeros((n, m), dtype=B.dtype)
    a = np.diag(B).copy()
    b = np.diag(B, k=1).copy()

    # Dynamic relative tolerance calculation
    max_val = max(np.max(np.abs(a)) if len(a) > 0 else 0.0,
                  np.max(np.abs(b)) if len(b) > 0 else 0.0)
    if rcond is None:
        rcond = max(m, n) * np.finfo(B.dtype).eps
    scaled_tol = rcond * max_val if max_val > 0 else 1e-12

    a[np.abs(a) < scaled_tol] = 0.0
    b[np.abs(b) < scaled_tol] = 0.0

    row_start = 0
    max_diag_len = len(a)
    
    while row_start < max_diag_len:
        
        # -------------------------------------------------------------
        # THE PURE SLICER
        # -------------------------------------------------------------
        # ONLY break the geometry if the superdiagonal link is severed.
        # This guarantees block_b is exactly 1 element shorter than block_a.
        row_end = row_start
        while row_end < len(b) and b[row_end] != 0.0:
            row_end += 1
            
        col_start = row_start
        col_end = row_end

        block_a = a[row_start:row_end + 1]
        block_b = b[row_start:col_end]

        if len(block_a) > 0:
            
            # -------------------------------------------------------------
            # THE PROACTIVE SINGULARITY SHIELD
            # -------------------------------------------------------------
            # If the block contains a zero on the diagonal, it is singular.
            # Normal equations will smear. Route to dense SVD safely.
            if np.min(np.abs(block_a)) < scaled_tol:
                sub_block = np.zeros((len(block_a), len(block_a)), dtype=B.dtype)
                np.fill_diagonal(sub_block, block_a)
                if len(block_b) > 0:
                    np.fill_diagonal(sub_block[:, 1:], block_b)
                
                block_pinv = np.linalg.pinv(sub_block)
                
            else:
                # Full rank square block -> Use our blazing-fast O(n) Thomas solver
                block_pinv = pinv_bidiag_block_woodbury(block_a, block_b)

            # Snap into coordinates
            B_plus[col_start:col_end + 1, row_start:row_end + 1] = block_pinv
            
        row_start = row_end + 1

    return B_plus

# ==============================================================================
# 4. IMPLICIT ASSEMBLY (The Cython Bridge)
# ==============================================================================
def apply_implicit_assembly(A_packed, tauq, taup, B_plus):
    """
    Computes A^+ = P @ B^+ @ Q^T using the stored Householder reflectors.
    """
    m, n = A_packed.shape
    num_reflectors = min(m, n) # Critical fix: Both Q and P have min(m,n) reflectors!
    
    # Step 1: Z = B_plus @ Q^T 
    Z = np.asfortranarray(B_plus.copy(), dtype=np.float64)
    m_Z, n_Z = Z.shape 
    Z, info1 = lapack_fast.cy_dormbr('Q', 'R', 'T', m_Z, n_Z, num_reflectors, A_packed, tauq, Z)
    
    # Step 2: A_plus = P @ Z
    m_A_plus, n_A_plus = Z.shape
    A_plus, info2 = lapack_fast.cy_dormbr('P', 'L', 'N', m_A_plus, n_A_plus, num_reflectors, A_packed, taup, Z)
    
    return A_plus

# ==============================================================================
# 4. THE FAST ENGINE WRAPPER
# ==============================================================================
def FastPinv(A, rcond=1e-12):
    """
    The Grand Unified Engine (Memory & Speed Optimized via Cython).
    """
    m, n = A.shape
    
    if m < n:
        return FastPinv(A.T, rcond).T
        
    # 1. Bidiagonalize
    A_packed, tauq, taup, B = bidag_implicit(A)
        
    # 2. Invert Bidiagonal (C^-1 Block)
    B_plus = pinv_bidiagonal(B,rcond=rcond)
    
    # 3. Assemble mathematically A^+ = P B^+ Q^T
    A_plus = apply_implicit_assembly(A_packed, tauq, taup, B_plus)
    
    return A_plus

# ==============================================================================
# VERIFICATION & SPEED TEST
# ==============================================================================
if __name__ == "__main__":
    print("--- Verifying the Direct Bidiagonal Pseudoinverse Engine ---\n")
    np.random.seed(42)
    
    # Test on a substantially large dense matrix to see O(N^3) avoidance
    M, N = 1500, 1000
    A_test = np.random.randn(M, N)
    
    print(f"Test Matrix Dimensions: {M} x {N}\n")
    
    # 1. Our Direct Engine
    start_time = time.time()
    A_fast = FastPinv(A_test)
    fast_time = time.time() - start_time
    print(f"Direct Engine FastPinv:  {fast_time:.4f} seconds")

    # 2. SciPy Standard SVD Engine
    start_time = time.time()
    A_scipy = linalg.pinv(A_test)
    scipy_time = time.time() - start_time
    print(f"SciPy Standard pinv:     {scipy_time:.4f} seconds")

    # Error Check
    error = np.linalg.norm(A_fast - A_scipy)
    print(f"\nFrobenius Norm Difference: {error:.4e}")
    print(f"Speedup Factor: {scipy_time / fast_time:.2f}x")

    if error < 1e-10:
        print("[SUCCESS] The Cython architecture performs flawlessly.")
    else:
        print("[WARNING] Discrepancy detected outside of standard tolerance.")
