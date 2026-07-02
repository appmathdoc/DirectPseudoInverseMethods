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
# 2. THE ZERO-CHASING / DIRECT SOLVER (B^+)
# ==============================================================================
def pinv_bidiagonal(B, tol=1e-12):
    """
    Computes the dense pseudoinverse B^+.
    If the matrix is well-conditioned, B is the unbroken 'C' block and we 
    use the ultra-fast solve_triangular (Back-substitution).
    If ill-conditioned, we tear the matrix (isolating K).
    """
    m, n = B.shape
    B_plus = np.zeros((n, m), dtype=B.dtype)
    k = min(m, n)
    
    # Extract the main diagonal
    d = np.diag(B)
    
    # Check if we have the pristine 'C' form (no null space structural zeros)
    if np.all(np.abs(d[:k]) > tol):
        # B is purely the C block. Inverse is explicitly upper triangular.
        # solve_triangular executes our exact back-substitution logic at C-speed!
        B_plus[:k, :k] = linalg.solve_triangular(B[:k, :k], np.eye(k), lower=False)
    else:
        # Fallback Block-Isolation (K-block presence). 
        # For this dense speed test, we use standard pinv on the torn block.
        B_plus[:k, :k] = linalg.pinv(B[:k, :k], rtol=tol)
        
    return B_plus

# ==============================================================================
# 3. IMPLICIT ASSEMBLY (The Cython Bridge)
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
def FastPinvA(A, rcond=1e-12):
    """
    The Grand Unified Engine (Memory & Speed Optimized via Cython).
    """
    m, n = A.shape
    
    if m < n:
        return FastPinvA(A.T, rcond).T
        
    # 1. Bidiagonalize
    A_packed, tauq, taup, B = bidag_implicit(A)
        
    # 2. Invert Bidiagonal (C^-1 Block)
    B_plus = pinv_bidiagonal(B, tol=rcond)
    
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
    A_fast = FastPinvA(A_test)
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