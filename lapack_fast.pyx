# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_lapack cimport dgebrd, dormbr

# Initialize NumPy C-API
cnp.import_array()

def cy_dgebrd(cnp.ndarray[cnp.float64_t, ndim=2, mode='fortran'] a, int lwork=-1):
    """
    Cython wrapper replicating scipy.linalg.lapack.dgebrd.
    Returns: (a, d, e, tauq, taup, info)
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int k = m if m < n else n  # min(m, n)
    cdef int lda = m
    
    # Allocate output vectors
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] d = np.zeros(k, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] e = np.zeros(max(1, k-1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] tauq = np.zeros(k, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] taup = np.zeros(k, dtype=np.float64)
    cdef int info = 0

    # Workspace query step
    cdef int lwork_query = -1
    cdef double work_query = 0.0
    
    dgebrd(&m, &n, &a[0,0], &lda, 
           &d[0], &e[0], &tauq[0], &taup[0], 
           &work_query, &lwork_query, &info)

    # Determine final workspace size
    if lwork < 0:
        lwork = <int>work_query
    if lwork < max(1, max(m, n)):
        lwork = max(1, max(m, n))
        
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] work = np.empty(lwork, dtype=np.float64)

    # Actual LAPACK math execution
    dgebrd(&m, &n, &a[0,0], &lda, 
           &d[0], &e[0], &tauq[0], &taup[0], 
           &work[0], &lwork, &info)

    return a, d, e, tauq, taup, info


def cy_dormbr(str vect, str side, str trans, int m, int n, int k,
              cnp.ndarray[cnp.float64_t, ndim=2, mode='fortran'] a,
              cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] tau,
              cnp.ndarray[cnp.float64_t, ndim=2, mode='fortran'] c, int lwork=-1):
    """
    Cython wrapper replicating scipy.linalg.lapack.dormbr.
    Returns: (c, info)
    """
    # Fortran expects char pointers. Extract the first character byte.
    cdef char vect_char = ord(vect[0])
    cdef char side_char = ord(side[0])
    cdef char trans_char = ord(trans[0])

    cdef int lda = a.shape[0]
    cdef int ldc = c.shape[0]
    cdef int info = 0

    # Workspace query step
    cdef int lwork_query = -1
    cdef double work_query = 0.0

    # Notice the 14 arguments! (vect_char is now included)
    dormbr(&vect_char, &side_char, &trans_char, &m, &n, &k, 
           &a[0,0], &lda, 
           &tau[0], 
           &c[0,0], &ldc, 
           &work_query, &lwork_query, &info)

    # Determine final workspace size
    if lwork < 0:
        lwork = <int>work_query
        
    cdef int min_lwork = max(1, n) if side == 'L' else max(1, m)
    if lwork < min_lwork:
        lwork = min_lwork

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='fortran'] work = np.empty(lwork, dtype=np.float64)

    # Actual LAPACK math execution
    dormbr(&vect_char, &side_char, &trans_char, &m, &n, &k, 
           &a[0,0], &lda, 
           &tau[0], 
           &c[0,0], &ldc, 
           &work[0], &lwork, &info)

    return c, info