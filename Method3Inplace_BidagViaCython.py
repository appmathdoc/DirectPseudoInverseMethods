import numpy as np
from scipy import linalg
import lapack_fast
import time

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
# 2. 4-STEP DIRECT SUB-BLOCK SOLVER
# ==============================================================================

def direct_bidiagonal_4step_block_solver(sub_B):
    """
    Solves an isolated Upper Bidiagonal sub-block using the 4-step rank-1 projection.
    """
    m, n = sub_B.shape
    
    if m == 0 or n == 0:
        return np.zeros((n, m), dtype=sub_B.dtype)
    if m == 1 and n == 1:
        return np.array([[1.0 / sub_B[0, 0]]]) if sub_B[0, 0] != 0 else np.array([[0.0]])

    # -------------------------------------------------------------
    # THE PROACTIVE SINGULARITY SHIELD
    # -------------------------------------------------------------
    # If the minimum diagonal element is indistinguishable from numerical noise 
    # relative to the block, solve_triangular will explode (e.g. 1e15 coefficients).
    # We catch it early and safely route the block to the SVD pseudo-inverse.
    diag_R = np.diag(sub_B)
    max_val = np.max(np.abs(sub_B))
    
    if len(diag_R) > 0 and max_val > 0:
        if np.min(np.abs(diag_R)) < 1e-11 * max_val:
            return np.linalg.pinv(sub_B)

    # Case 1: Perfectly Square (Upper Triangular)
    if m == n:
        try:
            return linalg.solve_triangular(sub_B, np.eye(m), lower=False)
        except (linalg.LinAlgError, ValueError):
            return linalg.pinv(sub_B)

    # Case 2: Wide Rectangular (e.g., 2 x 3) -> Underdetermined
    elif m < n:
        R = sub_B[:, :m]
        a = sub_B[:, m:] 
        
        try:
            R_inv = scipy.linalg.solve_triangular(R, np.eye(m), lower=False)
        except (scipy.linalg.LinAlgError, ValueError):
            return np.linalg.pinv(sub_B)
        
        z = R_inv @ a
        z_flat = z.ravel()
        gamma = 1.0 / (1.0 + np.dot(z_flat, z_flat))
        
        z_zt = np.outer(z_flat, z_flat)
        top_block = R_inv - gamma * (z_zt @ R_inv)
        bottom_block = gamma * (z_flat.reshape(1, -1) @ R_inv)
        
        return np.vstack([top_block, bottom_block])

    # Tall fallback for structural safety.
    else:
        return np.linalg.pinv(sub_B)

# ==============================================================================
# 3. GEOMETRIC DECOUPLING ENGINE
# ==============================================================================

def pinv_bidiagonal(B, rcond=None):
    """
    Isolates sub-blocks from explicit dense B, constructs explicit local sub-matrices,
    and routes them through the 4-step solve_triangular architecture.
    """
    m, n = B.shape
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
        # We ONLY sever geometric sub-blocks when the superdiagonal link 
        # is broken (b[i] == 0). A zero on the diagonal (a[i] == 0) indicates 
        # a singular block, but it remains physically coupled to the column!
        row_end = row_start
        while row_end < len(b) and b[row_end] != 0.0:
            row_end += 1
            
        # Because we only sever at b[i], col_end is always equal to row_end,
        # guaranteeing strictly square coordinate extractions.
        col_start = row_start
        col_end = row_end

        sub_block = B[row_start:row_end + 1, col_start:col_end + 1]

        if sub_block.size > 0:
            block_pinv = direct_bidiagonal_4step_block_solver(sub_block)
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
def FastPinv4Step(A, rcond=1e-12):
    """
    The Grand Unified Engine (Memory & Speed Optimized via Cython).
    """
    m, n = A.shape
    
    if m < n:
        return FastPinv4Step(A.T, rcond).T
        
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
    A_fast = FastPinv4Step(A_test)
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
