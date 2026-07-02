import sympy as sp

def is_zero_sym(val):
    """Safely evaluates if a symbolic expression is exactly zero."""
    if val == 0 or val.is_zero: 
        return True
    return sp.simplify(val) == 0

def quarantine_deflation_pinv_sympy(B_orig):
    """
    Executes the 'Quarantine Zone' Givens deflation on an m x n matrix,
    reducing it to [[C, 0], [0, K]] form and returning B^+.
    """
    m, n = B_orig.shape
    N = max(m, n)
    
    # 1. Pad the matrix to N x N to make the null space geometrically explicit
    B = sp.zeros(N, N)
    B[:m, :n] = B_orig
    
    # Track the orthogonal basis geometries
    U_G = sp.eye(N)
    V_G = sp.eye(N)
    
    active_size = N
    i = active_size - 1
    
    # Bottom-Up Traversal
    while i >= 0:
        if is_zero_sym(B[i, i]):
            
            # ---------------------------------------------------------
            # A. CHASE RIGHT (Row Operations / Left Multiply)
            # Zeros out the row i to the right of the diagonal
            # ---------------------------------------------------------
            for j in range(i + 1, active_size):
                a = B[j, j]
                b = B[i, j]
                if is_zero_sym(b):
                    continue
                
                r = sp.sqrt(a**2 + b**2)
                c = a / r
                s = b / r
                
                # Apply rotation to rows i and j
                for col in range(i, active_size):
                    val_j, val_i = B[j, col], B[i, col]
                    B[j, col] = sp.simplify( c * val_j + s * val_i)
                    B[i, col] = sp.simplify(-s * val_j + c * val_i)
                    
                # Apply rotation to U_G columns i and j
                for row in range(N):
                    val_j, val_i = U_G[row, j], U_G[row, i]
                    U_G[row, j] = sp.simplify( c * val_j + s * val_i)
                    U_G[row, i] = sp.simplify(-s * val_j + c * val_i)
                    
            # ---------------------------------------------------------
            # B. CHASE UP (Column Operations / Right Multiply)
            # Zeros out column i above the diagonal
            # ---------------------------------------------------------
            for j in range(i - 1, -1, -1):
                a = B[j, j]
                b = B[j, i]
                if is_zero_sym(b):
                    continue
                
                r = sp.sqrt(a**2 + b**2)
                c = a / r
                s = b / r
                
                # Apply rotation to cols i and j
                for row in range(j + 1):
                    val_j, val_i = B[row, j], B[row, i]
                    B[row, j] = sp.simplify( c * val_j + s * val_i)
                    B[row, i] = sp.simplify(-s * val_j + c * val_i)
                    
                # Apply rotation to V_G cols i and j
                for row in range(N):
                    val_j, val_i = V_G[row, j], V_G[row, i]
                    V_G[row, j] = sp.simplify( c * val_j + s * val_i)
                    V_G[row, i] = sp.simplify(-s * val_j + c * val_i)
                    
            # ---------------------------------------------------------
            # C. THE IMMEDIATE QUARANTINE SHIFT
            # ---------------------------------------------------------
            target_idx = active_size - 1
            
            if i != target_idx:
                # Shift the zeroed row to the bottom of the active workspace
                row_order = list(range(N))
                row_order.insert(target_idx, row_order.pop(i))
                B = B[row_order, :]
                U_G = U_G[:, row_order]
                
                # Shift the zeroed column to the far right of the active workspace
                col_order = list(range(N))
                col_order.insert(target_idx, col_order.pop(i))
                B = B[:, col_order]
                V_G = V_G[:, col_order]
                
            # The dimension is severed and moved. Shrink the active workspace.
            active_size -= 1
            
        i -= 1

    # ---------------------------------------------------------
    # 3. CORE INVERSION & ASSEMBLY
    # ---------------------------------------------------------
    B_plus_work = sp.zeros(N, N)
    
    if active_size > 0:
        # Extract the pristine, guaranteed full-rank C block
        C = B[:active_size, :active_size]
        
        # Invert (SymPy handles this symbolically, but for numerical logic 
        # this is a simple O(n^2) back-substitution)
        B_plus_work[:active_size, :active_size] = C.inv()
        
    # Reassemble: B^+ = V_G * B_plus_work * U_G^T
    B_plus_full = V_G * B_plus_work * U_G.T
    
    # Slice to match the required pseudoinverse dimensions (n x m)
    B_plus_final = B_plus_full[:n, :m]
    
    return B_plus_final

if __name__ == '__main__': 
    x, y, z = sp.symbols('x y z', positive = True)
    
    print("\n--- SYMPY EXACT TEST ---")
    
    # A 6x7 bidiagonal matrix with mix of symbolic and numerical 
    B_sym = sp.Matrix([
        [y,  0,  0,  0, 0, 0, 0],
        [0,  1,  3,  0, 0, 0, 0],
        [0,  0,  1,  2, 0, 0, 0],
        [0,  0,  0,  0, 3, 0, 0],
        [0,  0,  0,  0, 1, x, 0], 
        [0,  0,  0,  0, 0, 1, 2]
    ])

    
    print("Original Symbolic Bidiagonal Matrix B:")
    print(B_sym)  
    
    B_pinv = quarantine_deflation_pinv_sympy(B_sym)
    B_pinv = sp.simplify(sp.together(B_pinv))
    
    print("\nExact Symbolic MP3 Pseudoinverse:")
    print(B_pinv)

    # We can verify Condition 1: B * B_pinv * B == B
    # (Note: Symbolic simplification can take a moment)
    print("\nVerifying Penrose Condition 1 (B * B^+ * B == B):")
    cond_1 = sp.simplify(B_sym * B_pinv * B_sym)
    print(cond_1)
    print("Is Condition 1 met?", cond_1 == B_sym)