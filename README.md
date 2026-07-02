# DirectPseudoInverseMethods
Numpy/Scipy implementations of algorithms for direct computation of the Moore Penrose Pseudoinverse

Three Methods for Direct Computation of the Moore Penrose Pseudoinverse (i.e., does not use SVD). These algorithms were designed to 
* avoid dense inverse matrices and thus dense MP Pseudoinverses (by mapping inverses to O(n^2) back substitution implementations)
* be as fast or faster than traditional SVD based methods
All three methods rely on Bidiagonalization (either Golub-Kahan or Lanczos) to reduce the computation to a bidiagonal matrix.  The three Moore Penrose bidiagonal pseudo-inverse methods are 
* __The CK Decomposition:__ If $B$ is upper bidiagonal, then there exists upper _invertible_ bidiagonal $C$ and a superdiagonal matrix $K$ (only non-zero coefficients are on the superdiagonal) such that $$B = U \begin{bmatrix} C & 0 \\ 0 & K \end{bmatrix}$$
* __Dual Normal Equations:__ The Thomas algorithm and Woodbury tearing are applied to the dual normal equations of $BB^T$.
* __In Place Pseudo-inverse:__ Splitting $B$ into block diagonal form on the coefficients $\beta_{j}=0$ and then splitting each block at $\alpha_k = 0$ into block matrices with zeros off the block diagonal and a mixture of square and rectangular matrices on the block's diagonal.

Method 0 is simply the application of scipy.linalg.pinv to the bidiagonal and is included for comparison.  Method 1 is implemented for bidiagonal __sympy__ matrices$^\ast$.  Method 2 is dual normalization.  Method 3 has two versions.  The first does not require gebrd/orgbr and is the one tested in Method3Tests.py.  The second Method 3 is Inplace with Cython wrapping to gebrd/orbgr.  This latter approach requires compiling the __lapack_fast.pyx__ cython file by executing __lapack_fast_compile_setup.py__.  That is, for __lapack_fast.pyx__ in the same directory, run
```
python lapack_fast_compile_setup.py
```
__Note:__ Requires C runtime tools. 

The speed tests are included simply to demonstrate that these 3 methods are _no slower_ in general than the usual SVD pinv. Significant speedups would require implementations that do not rely on the Python interpreter as much as these implementations do. 

$^\ast$ Sympy uses the MacDuffee formula to calculate pinv, which for a computer algebra system is more than sufficient.  We included the CK decomposition here not so much as a tool for calculating pseudo-inverses but as a tool for producing symbolic CK decompsitions.  
