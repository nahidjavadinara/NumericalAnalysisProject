# Numerical Analysis Project
## Resolution of a 2D Poisson problem 
### Introduction

This section outlines the development and validation of a second-order finite difference solver for the 2D Poisson equation. The solver discretizes the domain, formulates the discrete Laplacian, assembles the corresponding linear system, and validates the implementation against an exact solution.

---


### Objectives 

**Finite Difference Discretization:** Derive and implement a discrete version of the 2D Poisson equation, validate the solver using a known exact solution, and analyze convergence and computational efficiency.

**Linear System Solvers:** Explore both direct and iterative methods for solving the resulting linear system, comparing computational costs and performance.

**Extensions and Enhancements:** Incorporate additional features such as diffusion terms, higher-order finite difference schemes, and advanced boundary conditions.


## Solving the Poisson Equation Using Finite Difference Method

### Finite Difference Discretization

The Poisson equation in two dimensions is:

$$
-\Delta u = f(x, y), \quad \Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}.
$$

Using finite difference approximation, the second derivatives are approximated as:

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i-1,j} - 2u_{i,j} + u_{i+1,j}}{\Delta x^2}, \quad
\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i,j-1} - 2u_{i,j} + u_{i,j+1}}{\Delta y^2}.
$$

Substituting these into the Poisson equation gives:

$$
-\left(\frac{u_{i-1,j} - 2u_{i,j} + u_{i+1,j}}{\Delta x^2} + \frac{u_{i,j-1} - 2u_{i,j} + u_{i,j+1}}{\Delta y^2}\right) = f(x_i, y_j).
$$

For a uniform grid $$(\(\Delta x = \Delta y = h\))$$:

$$
-\frac{1}{h^2} \left(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}\right) = f(x_i, y_j).
$$

***Import Necessary Libraries***
```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
```
## 1.1 Designing the Solver

### 1.1.1 Generating a Rectangular Grid

**Objective:**

Create a uniform rectangular grid of $(N_x - 1) * (N_y - 1)$ points representing the discretization of the interior domain $\Omega$. Each grid point $(x_i, y_j)$ has indices $i \in \{1, \dots, N_x - 1\}$ and $j \in \{1, \dots, N_y - 1\}$.

***Define the domain and grid***
```python
Lx, Ly = 1.0, 1.0  # Domain size
grid_number = 10    # From [10, 20, 40, 80]
Nx, Ny = grid_number, grid_number  
hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
```
***Generate grid***
```python
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
```
***Construct the sparse matrix A*** 
```python
main_diag = -4 * np.ones((Nx - 2) * (Ny - 2))
off_diag = np.ones((Nx - 2) * (Ny - 2) - 1)
off_diag2 = np.ones((Nx - 2) * (Ny - 2) - (Nx - 2))
```
***Create diagonals***
```python
diagonals = [main_diag, off_diag, off_diag, off_diag2, off_diag2]
offsets = [0, 1, -1, Nx - 2, -(Nx - 2)]
A = diags(diagonals, offsets, format='csr') / hx**2
```

***Solve the system*** 
```python
u_flat = spsolve(A, f_flat)
```
***Reshape solution back into 2D***
```python
u = np.zeros((Nx, Ny))
u[1:-1, 1:-1] = u_flat.reshape((Nx - 2, Ny - 2))
```
***Plot the solution***
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_title("Solution to the Poisson Equation")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
```


### 1.2 Validation of the Implementation

To validate the implementation, we define an exact solution:

$$
u_{\text{ex}}(x, y) = \sin^2(\pi x) \sin^2(\pi y)
$$

### Finding the Right-Hand Side $$\(f(x, y)\)$$

Taking the Laplacian of $$\(u_{\text{ex}}(x, y)\)$$:

$$
\Delta u_{\text{ex}} = \frac{\partial^2 u_{\text{ex}}}{\partial x^2} + \frac{\partial^2 u_{\text{ex}}}{\partial y^2}.
$$

Compute the partial derivatives:

1. First derivative concerning $$\(x\)$$:
   
$$
\frac{\partial u_{\text{ex}}}{\partial x} = 2\sin(\pi x)\cos(\pi x)\sin^2(\pi y).
$$

2. Second derivative with respect to $$\(x\)$$:

$$
\frac{\partial^2 u_{\text{ex}}}{\partial x^2} = 2\pi^2\cos(2\pi x)\sin^2(\pi y).
$$

3. Similarly, for $$\(y\)$$:

$$
\frac{\partial^2 u_{\text{ex}}}{\partial y^2} = 2\pi^2\cos(2\pi y)\sin^2(\pi x).
$$

Substituting into $$\(\Delta u_{\text{ex}}\)$$ gives:

$$
f(x, y) = -\Delta u_{\text{ex}} = -2\pi^2\left(\cos(2\pi x)\sin^2(\pi y) + \cos(2\pi y)\sin^2(\pi x)\right).
$$

***Define the source term f(x, y)***
```python
f = 2 * np.pi**2 * ((np.sin(np.pi * Y))**2 * np.cos(2 * np.pi * X) + (np.sin(np.pi * X))**2 * np.cos(2 * np.pi * Y))
```

***Flatten the source term into a 1D vector***
```python
b = new_f[1:-1, 1:-1].flatten()
```


### Consistency with Boundary Conditions

The solution $$\(u_{\text{ex}}(x, y)\)$$ satisfies the homogeneous Dirichlet boundary conditions:

$$
u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0.
$$

#### Apply boundary conditions (homogeneous Dirichlet)
```python
u_boundary = np.zeros_like(x)
    u_boundary[0, :] = 0      # Bottom boundary
    u_boundary[-1, :] = 0    # Top boundary
    u_boundary[:, 0] = 0      # Left boundary
    u_boundary[:, -1] = 0    # Right boundary
```
#### Extend the procedure for a non-zero boundary condition
```python
b_matrix = np.zeros_like(X)  # Initialize the matrix for boundary contributions
b_matrix[1:-1, 1:-1] = f[1:-1, 1:-1]  # Add the source term for interior points
u_boundry = boundary_condition(X,Y)   #for non-zero boundary condition
new_f = b_matrix + u_boundry      # Au = b + f
b = new_f[1:-1, 1:-1].flatten()
```

### Numerical Validation

#### Relative Error and Convergence

Using a grid with $$\(N \times N\)$$ points, solve the linear system (by putting the previous steps in a function) for the numerical solution $$\(u_{\text{num}}\)$$ and compute the relative error in the maximum norm:

$$
\text{Error} = \max_{i,j} \left| \frac{u_{\text{num}}(x_i, y_j) - u_{\text{ex}}(x_i, y_j)}{u_{\text{ex}}(x_i, y_j) + \epsilon} \right|
$$

where $$\( \epsilon = 10^{-12} \)$$ is a small constant to avoid division by zero.


#### Convergence Plot

Generate a log-log plot of the error against the step size $$\(h = \frac{1}{N}\)$$. The expected convergence rate for the second-order finite difference scheme is $$\(O(h^2)\)$$.

```python

# Define step sizes and errors
grids = [10, 20, 40, 80]
errors = []
step_sizes = []
for grid in grids:
    X, Y , u = solver(grid)
    u_exact = exact_solution(X,Y)
    error = np.max(np.abs((u - u_exact) / (u_exact + 1e-12)))  # Maximum norm error
    errors.append(error)
    step_sizes.append(1/grid)

# Plot
plt.figure()
plt.loglog(step_sizes, errors, marker='o', label='Numerical Error')
plt.title('Convergence of Finite Difference Method')
plt.xlabel('Step size (h)')
plt.ylabel('Error (log scale)')
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()
plt.show()
```
# 2. Solving the Linear System

This section explores various methods to solve the linear system arising from the discretization of the 2D Poisson equation. We compare **direct** and **iterative** methods in terms of computational cost and efficiency.

---

## 2.1 Direct Methods

### 2.1.1 Sparse Matrix Representation

#### Explanation:

In the initial implementation, a **dense matrix representation** was used for the discrete Laplacian, storing all matrix entries explicitly. However, the discrete Laplacian matrix $A$ is **sparse**, meaning that most of its elements are zero. Specifically, for a grid with $N_x \times N_y$ interior points, each row in $A$ has at most five non-zero entries: one on the main diagonal and up to four on the neighboring diagonals (left, right, top, bottom).

#### Advantages of Sparse Representation:

1. **Memory Efficiency**: Sparse matrices require significantly less memory by storing only non-zero elements.
2. **Computational Efficiency**: Operations on sparse matrices (e.g., matrix-vector multiplication, solving linear systems) are faster due to the reduced number of computations.
3. **Scalability**: Sparse representations are essential for large-scale problems where dense matrices become infeasible.

```python
import scipy.sparse as sp
def assemble_sparse_matrix_direct(Nx, Ny, h):
    """
    Assembles the sparse matrix A for the discrete Poisson equation using Kronecker products.

    Parameters:
    - Nx, Ny: Number of interior points along x and y axes.
    - h: Grid spacing (assuming h_x = h_y = h).

    Returns:
    - A: Sparse matrix in CSR format.
    """
    # 1D Laplacian matrix T (tridiagonal)
    main_diag = 4.0 * np.ones(Nx)
    off_diag = -1.0 * np.ones(Nx - 1)
    diagonals = [off_diag, main_diag, off_diag]
    T = sp.diags(diagonals, offsets=[-1, 0, 1], format='csr')

    # Identity matrix
    I = sp.eye(Nx, format='csr')

    # 2D Laplacian using Kronecker products
    A = sp.kron(I, T) + sp.kron(T, I)

    # Scale by 1/h^2
    A /= h**2

    return A
```
### 2.1.2. Computational Time Comparison

Empirically compare the computational time between sparse and dense solvers by performing experiments with increasing grid sizes.
```python
def solve_direct_sparse(Nx, Ny, f_func, boundary_func):
    """
    Solves the Poisson equation using a sparse direct solver.

    Parameters:
    - Nx, Ny: Number of interior points along x and y axes.
    - f_func: Source term function f(x, y).
    - boundary_func: Boundary condition function g(x, y).

    Returns:
    - u_numeric: 2D array of numerical solution values.
    - solve_time: Time taken to solve.
    """
    # Generate grid
    x, y, X, Y, h_x, h_y = generate_grid(Nx, Ny, Lx=1.0, Ly=1.0)

    # Assemble RHS vector with boundary conditions
    b = assemble_rhs_direct(Nx, Ny, X, Y, f_func, boundary_func, h_x)

    # Assemble Sparse Matrix
    A_sparse = assemble_sparse_matrix_direct(Nx, Ny, h_x)

    # Solve the linear system and time the operation
    start_time = time.time()
    u = spsolve(A_sparse, b)
    end_time = time.time()
    solve_time = end_time - start_time

    # Reshape solution to 2D grid
    u_numeric = u.reshape((Nx, Ny), order='F')  # Column-major

    return u_numeric, solve_time
```
### 2.1.3 Advantages of Sparse Matrix Representation

#### Why Sparse Matrices are Advantageous:

1. **Reduced Memory Consumption:**
   - **Dense Matrix**: Requires $\mathcal{O}(N^2)$ memory for a matrix of size $N \times N$.
   - **Sparse Matrix**: Requires $\mathcal{O}(N)$ memory for matrices with a constant number of non-zero entries per row.

   For large $N$, the memory savings are substantial.

2. **Faster Computations:**
   - **Matrix-Vector Multiplication:**
     - **Dense**: $\mathcal{O}(N^2)$ operations.
     - **Sparse**: $\mathcal{O}(N)$ operations, assuming a constant number of non-zero entries per row.
   - **Solving Linear Systems:**
     - **Gaussian Elimination:**
       - **Dense**: $\mathcal{O}(N^3)$ operations.
       - **Sparse**: For tridiagonal matrices, Gaussian elimination requires $\mathcal{O}(N)$ operations.
   - **Iterative Methods:**
     - Benefit from efficient sparse matrix operations, leading to faster convergence.

3. **Scalability:**
   - Sparse matrices enable the solution of much larger systems that would be infeasible with dense matrices due to memory and computational constraints.

---

### Justification via Operation Counts:

- **Gaussian Elimination for Tridiagonal Matrix:**
  - **Dense Matrix:**
    - Requires $\mathcal{O}(N^3)$ operations.
  - **Tridiagonal Matrix:**
    - Utilizes the **Thomas algorithm**, an optimized form of Gaussian elimination.
    - Requires only $\mathcal{O}(N)$ operations.

#### Conclusion:
Sparse representations dramatically reduce computational costs, especially for structured matrices like tridiagonal.

### Visualization of Operation Counts:

| Method                     | Dense Matrix       | Tridiagonal Sparse Matrix |
|----------------------------|--------------------|---------------------------|
| Gaussian Elimination       | $\mathcal{O}(N^3)$ | $\mathcal{O}(N)$          |
| Matrix-Vector Multiplication | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$          |
| Memory Usage               | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$          |


## 2.2 Iterative Methods

This section investigates the performance of various iterative methods for solving the linear system derived from the discretization of the $2D$ Poisson equation.  
We explore the **Jacobi**, **Gauss-Seidel**, and **Successive Over-Relaxation (SOR)** methods.

---

### 2.2.1 Eigenvalues of the Discrete Laplacian in $2D$

#### Explanation:
Understanding the eigenvalues of the discrete Laplacian matrix $A$ is crucial for analyzing the convergence properties of iterative solvers.

#### 1. Eigenvalues of Matrix $T$ in $1D$:
- For a tridiagonal matrix $T$ representing the 1D Laplacian with Dirichlet boundary conditions:

$$
T =
\begin{bmatrix}
4 & -1 & 0 & \cdots & 0 \\
-1 & 4 & -1 & \ddots & 0 \\
0 & -1 & 4 & \ddots & \vdots \\
\vdots & \ddots & \ddots & \ddots & -1 \\
0 & \cdots & 0 & -1 & 4
\end{bmatrix}
$$

- **Eigenvalues**:

$$
\lambda_i = 2 - 2 \cos \left(\frac{\pi i}{N + 1}\right), \quad i = 1, 2, \dots, N
$$

- **Derivation**:
  The eigenvalues are derived based on the structure of $T$ by utilizing recurrence relations and properties of Chebyshev polynomials.

---

#### 2. Eigenvalues of the 2D Discrete Laplacian Using Kronecker Products:
- Given $A = I \otimes T + T \otimes I$, where $I$ is the identity matrix and $T$ is the 1D Laplacian matrix.

- **Eigenvalues of $A$**:

$$
\lambda_{i,j} = \lambda_i + \lambda_j = 4 - 2 \cos \left(\frac{\pi i}{N_x + 1}\right) - 2 \cos \left(\frac{\pi j}{N_y + 1}\right),
\quad i = 1, 2, \dots, N_x; \quad j = 1, 2, \dots, N_y
$$

- **Implications**:
  The eigenvalues of $A$ influence the convergence rate of iterative methods.  
  Specifically, methods like **Jacobi** and **Gauss-Seidel** converge faster when the spectral radius of their iteration matrices is smaller.

### 2.2.2. Implementation of Iterative Methods

**Jacobi Method Implementation**
```python
def jacobi_method(A, b, tol=1e-6, max_iterations=1000):
    """
    Implements the Jacobi iterative method.

    Parameters:
    - A: Sparse matrix A in CSR format.
    - b: RHS vector.
    - tol: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.

    Returns:
    - u: Solution vector.
    - iterations: Number of iterations performed.
    - residuals: List of residual norms at each iteration.
    """
    D = A.diagonal()
    R = A - sp.diags(D, format='csr')
    u = np.zeros_like(b)
    residuals = []

    for iteration in range(max_iterations):
        u_new = (b - R.dot(u)) / D
        residual = np.linalg.norm(b - A.dot(u_new), ord=np.inf)
        residuals.append(residual)

        if residual < tol:
            return u_new, iteration + 1, residuals

        u = u_new.copy()

    print("Jacobi method did not converge within the maximum number of iterations.")
    return u, iteration + 1, residuals
```
**Gauss-Seidel Method Implementation**
```python
def gauss_seidel_method(A, b, tol=1e-6, max_iterations=1000):
    """
    Implements the Gauss-Seidel iterative method.

    Parameters:
    - A: Sparse matrix A in CSR format.
    - b: RHS vector.
    - tol: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.

    Returns:
    - u: Solution vector.
    - iterations: Number of iterations performed.
    - residuals: List of residual norms at each iteration.
    """
    A_csr = A.tocsr()
    N = len(b)
    u = np.zeros_like(b)
    residuals = []

    for iteration in range(max_iterations):
        for i in range(N):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            Ai = A_csr.indices[row_start:row_end]
            Av = A_csr.data[row_start:row_end]

            sigma = 0.0
            for idx, j in enumerate(Ai):
                if j != i:
                    sigma += Av[idx] * u[j]
            u[i] = (b[i] - sigma) / Av[Ai == i][0]

        residual = np.linalg.norm(b - A.dot(u), ord=np.inf)
        residuals.append(residual)

        if residual < tol:
            return u, iteration + 1, residuals

    print("Gauss-Seidel method did not converge within the maximum number of iterations.")
    return u, iteration + 1, residuals
```

### 2.2.3 Convergence Expectations

#### Convergence of Iterative Methods:
For iterative methods like **Jacobi** and **Gauss-Seidel**, convergence is influenced by the properties of the coefficient matrix $A$.

#### Criteria for Convergence:

1. **Spectral Radius ($\rho$) of Iteration Matrix:**
   - For both Jacobi and Gauss-Seidel methods, the iteration matrix $M$ determines convergence.
   - **Convergence Condition:** $\rho(M) < 1$.

2. **Matrix Properties:**
   - **Strict Diagonal Dominance:**
     - A matrix $A$ is strictly diagonally dominant if for each row $i$:

$$
|A_{ii}| > \sum_{j \neq i} |A_{ij}|
$$

   - **Implication:** Strict diagonal dominance guarantees convergence of both Jacobi and Gauss-Seidel methods.
   
   - **Symmetry and Positive Definiteness:**
     - **Symmetric Positive Definite (SPD)** matrices also ensure convergence.

3. **Effect of Grid Size:**
   - As the number of grid points increases, the condition number of $A$ may worsen, potentially affecting the convergence rate.

---

#### Analysis for the Discrete Poisson Problem:

- The **discrete Laplacian matrix** $A$ is **symmetric and positive definite**.
- It is also **strictly diagonally dominant**.
- **Conclusion:** Both Jacobi and Gauss-Seidel methods are **guaranteed to converge** for the discrete Poisson problem.

#### Impact of Increasing Grid Points:

- As $N$ increases:
  - The grid spacing $h$ decreases, leading to a larger condition number for $A$.
  - **Implication:** The convergence rate may slow down as the grid becomes finer.

### 2.2.4 Measuring the Cost of Iterative Solvers

#### Components of Cost Measurement:

1. **Number of Iterations:**
   - Indicates how quickly the method converges to the desired tolerance.
   - Fewer iterations imply faster convergence.

2. **Cost per Iteration:**
   - Involves operations such as matrix-vector multiplications, vector updates, and residual computations.
   - Dependent on the sparsity and size of matrix $A$.

3. **Overall Computational Time:**
   - Combination of the above two factors.
   - Represents the total time taken to reach convergence.

---

#### Metrics to Measure:

- **Iterations to Convergence:**
  - The total number of iterations needed for the residual to fall below a predefined tolerance.

- **Residual Reduction:**
  - Tracking how the residual decreases with each iteration provides insight into the convergence behavior.

- **Computational Time:**
  - Measuring the total elapsed time from the start of the iterations until convergence.

---

#### Implementation Considerations:

- **Residual Computation:**
  - Efficiently compute the residual $r = b - A u$ at each iteration.
  - Utilize norm calculations (e.g., infinity norm) to assess convergence.

- **Stopping Criteria:**
  - Defined based on the residual norm and a tolerance level (e.g., $1 \times 10^{-6}$).

- **Recording Metrics:**
  - Store the residuals and iteration counts for analysis and plotting.

### 2.2.5 Successive Over-Relaxation (SOR) Method

#### Explanation:
The **Successive Over-Relaxation (SOR)** method enhances the **Gauss-Seidel** approach by introducing a relaxation parameter $\omega$ to accelerate convergence potentially.

- **Update Formula:**

$$
u_i^{(k+1)} = (1 - \omega) u_i^{(k)} + \frac{\omega}{A_{ii}} \left( b_i - \sum_{j<i} A_{ij} u_j^{(k+1)} - \sum_{j>i} A_{ij} u_j^{(k)} \right)
$$

- **Relaxation Parameter $\omega$:**
  - **Range:** $0 < \omega < 2$ ensures convergence.
  - **Optimal Value:** The value of $\omega$ that minimizes the number of iterations required for convergence.

  - **Effects:**
    - **$\omega = 1$**: Reduces to the Gauss-Seidel method.
    - **$\omega > 1$**: Over-relaxation; can lead to faster convergence.
    - **$\omega < 1$**: Under-relaxation; can dampen oscillations but may slow convergence.

```python
def sor_method(A, b, omega, tol=1e-6, max_iterations=1000):
    """
    Implements the Successive Over-Relaxation (SOR) method.

    Parameters:
    - A: Sparse matrix A in CSR format.
    - b: RHS vector.
    - omega: Relaxation parameter.
    - tol: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.

    Returns:
    - u: Solution vector.
    - iterations: Number of iterations performed.
    - residuals: List of residual norms at each iteration.
    """
    A_csr = A.tocsr()
    N = len(b)
    u = np.zeros_like(b)
    residuals = []

    for iteration in range(max_iterations):
        for i in range(N):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            Ai = A_csr.indices[row_start:row_end]
            Av = A_csr.data[row_start:row_end]

            sigma = 0.0
            for idx, j in enumerate(Ai):
                if j != i:
                    sigma += Av[idx] * u[j]
            u_new = (1 - omega) * u[i] + (omega / Av[Ai == i][0]) * (b[i] - sigma)
            u[i] = u_new

        residual = np.linalg.norm(b - A.dot(u), ord=np.inf)
        residuals.append(residual)

        if residual < tol:
            return u, iteration + 1, residuals

    print("SOR method did not converge within the maximum number of iterations.")
    return u, iteration + 1, residuals
```
**Finding the Optimal Relaxation Parameter ($ω$)**
```python
def find_optimal_omega(A, b, omegas, tol=1e-6, max_iterations=1000):
    """
    Finds the optimal relaxation parameter omega for the SOR method.

    Parameters:
    - A: Sparse matrix A in CSR format.
    - b: RHS vector.
    - omegas: List or array of omega values to test.
    - tol: Tolerance for convergence.
    - max_iterations: Maximum number of iterations.

    Returns:
    - optimal_omega: Omega with the least number of iterations.
    - omega_iterations: Dictionary mapping omega to iterations.
    """
    omega_iterations = {}
    for omega in omegas:
        _, iter_count, _ = sor_method(A, b, omega, tol, max_iterations)
        omega_iterations[omega] = iter_count
        print(f"Omega: {omega:.2f}, Iterations: {iter_count}")

    # Find omega with minimum iterations
    optimal_omega = min(omega_iterations, key=omega_iterations.get)
    return optimal_omega, omega_iterations
```
# 3. Extensions to the Solver

Building upon the foundational solver for the 2D Poisson equation, this section explores enhancing its accuracy by implementing a fourth-order finite difference formula. Higher-order schemes offer improved precision by reducing discretization errors, which is particularly beneficial for problems requiring stringent accuracy levels.

## 3.1 Fourth-Order Finite Difference Scheme

### 3.1.1 Motivation for Higher-Order Schemes

#### Second-Order Schemes:
- **Accuracy:** The second-order finite difference scheme approximates derivatives with a truncation error of $\mathcal{O}(h^2)$, where $h$ is the grid spacing.
- **Limitations:** While sufficiently accurate for many applications, certain problems may demand higher precision, especially when dealing with smooth solutions or when minimizing numerical artifacts is crucial.

#### Fourth-Order Schemes:
- **Enhanced Accuracy:** Reduces the truncation error to $\mathcal{O}(h^4)$, providing more accurate derivative approximations.
- **Sharper Gradients:** Better captures rapid changes in the solution without requiring excessively fine grids.
- **Trade-Off:** Increased stencil width leads to more complex matrix structures and slightly higher computational costs per iteration.

---

### 3.1.2 Derivation of the Fourth-Order Finite Difference Approximation

 **Objective:**
Derive a fourth-order accurate finite difference approximation for the Laplacian $\Delta u$ in 2D.

 **Approach:**

#### 1. Fourth-Order Central Difference in 1D:
The standard second-order central difference for the second derivative is:

$$
\frac{d^2 u}{dx^2} \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{h^2}
$$

To achieve fourth-order accuracy, incorporate additional points:

$$
\frac{d^2 u}{dx^2} \approx \frac{-u_{i-2} + 16u_{i-1} - 30u_i + 16u_{i+1} - u_{i+2}}{12h^2} + \mathcal{O}(h^4)
$$

---

#### 2. Extension to 2D:
Applying the fourth-order central difference in both $x$ and $y$ directions:

$$
\Delta_h u_{i,j} = \frac{-u_{i-2,j} + 16u_{i-1,j} - 30u_{i,j} + 16u_{i+1,j} - u_{i+2,j}}{12h_x^2}
+ \frac{-u_{i,j-2} + 16u_{i,j-1} - 30u_{i,j} + 16u_{i,j+1} - u_{i,j+2}}{12h_y^2}
$$

---

### **Uniform Grid Simplification:**
Assuming $h_x = h_y = h$, the discrete Laplacian simplifies to:

$$
\Delta_h u_{i,j} = \frac{-u_{i-2,j} + 16u_{i-1,j} - 30u_{i,j} + 16u_{i+1,j} - u_{i+2,j}}{12h^2}
+ \frac{-u_{i,j-2} + 16u_{i,j-1} - 30u_{i,j} + 16u_{i,j+1} - u_{i,j+2}}{12h^2}
$$

---

### **Final Discrete Laplacian:**

$$
\Delta_h u_{i,j} = \frac{-u_{i-2,j} + 16u_{i-1,j} - 30u_{i,j} + 16u_{i+1,j} - u_{i+2,j}}{12h^2}
+ \frac{-u_{i,j-2} + 16u_{i,j-1} - 30u_{i,j} + 16u_{i,j+1} - u_{i,j+2}}{12h^2}
$$

### 3.1.3 Implementing the Fourth-Order Finite Difference Scheme

#### 3.1.3.1 Grid Generation for Higher-Order Schemes

Higher-order schemes require additional grid points to compute derivatives. Specifically, the fourth-order Laplacian stencil extends **two points** in each direction, necessitating a **wider boundary layer**.

##### Considerations:
- **Boundary Points:** To accommodate the stencil, the grid must include additional points or implement **ghost points**.
- **Handling Boundaries:** Implementing boundary conditions becomes more intricate due to the extended stencil.

##### Implementation:
The existing `generate_grid` function can be adapted to account for the extended stencil by ensuring that the grid includes enough points to compute the necessary derivatives without accessing out-of-bound indices.

> **Note:** For simplicity, assume that boundary conditions are homogeneous Dirichlet ($u = 0$).

---

#### 3.1.3.2 Assembling the Fourth-Order Sparse Matrix

##### Key Differences from Second-Order Scheme:
- **Stencil Width:** Each row in the matrix now includes connections to points up to **two steps away** in both $x$ and $y$ directions.
- **Matrix Structure:** The matrix becomes **block pentadiagonal** instead of **block tridiagonal**, increasing the number of non-zero entries per row.

```python
def assemble_sparse_matrix_fourth_order(Nx, Ny, h):
    """
    Assembles the sparse matrix A for the discrete Poisson equation using fourth-order finite differences.

    Parameters:
    - Nx, Ny: Number of interior points along x and y axes.
    - h: Grid spacing (assuming h_x = h_y = h).

    Returns:
    - A: Sparse matrix in CSR format.
    """
    # 1D Fourth-Order Laplacian matrix T
    main_diag = 30.0 * np.ones(Nx)
    first_off_diag = 16.0 * np.ones(Nx - 1)
    second_off_diag = -1.0 * np.ones(Nx - 2)
    diagonals = [second_off_diag, first_off_diag, main_diag, first_off_diag, second_off_diag]
    offsets = [-2, -1, 0, 1, 2]
    T = sp.diags(diagonals, offsets, shape=(Nx, Nx), format='csr')

    # Identity matrix
    I = sp.eye(Nx, format='csr')

    # 2D Laplacian using Kronecker products
    A = sp.kron(I, T) + sp.kron(T, I)

    # Scale by 1/(12h^2)
    A /= (12 * h**2)

    return A
```
##### 3.1.3.3. Fourth-Order RHS Vector Assembly

Due to the wider stencil, points near the boundary require handling contributions from two steps away. For homogeneous Dirichlet conditions ($u=0$),these contributions are zero, simplifying the RHS vector.
```python
def assemble_rhs_fourth_order(Nx, Ny, X, Y, f_func, boundary_func, h):
    """
    Assembles the RHS vector b for the linear system Au = b using fourth-order finite differences, incorporating boundary conditions.

    Parameters:
    - Nx, Ny: Number of interior points along x and y axes.
    - X, Y: 2D meshgrid arrays of interior points.
    - f_func: Source term function f(x, y).
    - boundary_func: Boundary condition function g(x, y).
    - h: Grid spacing.

    Returns:
    - b: 1D RHS vector.
    """
    b = np.zeros(Nx * Ny)

    for j in range(Ny):
        for i in range(Nx):
            k = j * Nx + i  # Column-major ordering
            b[k] = f_func(X[i, j], Y[i, j])

            # Boundary contributions (homogeneous Dirichlet: u=0)
            # Since the stencil extends two points, check for two-step boundaries
            # Left boundary (i-2 < 0)
            if i < 2:
                # u_{i-2,j} = u_{i-2,j} = 0
                # u_{i-1,j} = u_{i-1,j} = 0
                b[k] += (-1.0 * boundary_func(0, Y[i, j]))  # u_{i-2,j}
                b[k] += (16.0 * boundary_func(0, Y[i, j]))  # u_{i-1,j}
            elif i == Nx - 2:
                # u_{i+2,j} = 0
                b[k] += (-1.0 * boundary_func(1, Y[i, j]))  # u_{i+2,j}
                b[k] += (16.0 * boundary_func(1, Y[i, j]))  # u_{i+1,j}

            # Bottom boundary (j-2 < 0)
            if j < 2:
                # u_{i,j-2} = 0
                # u_{i,j-1} = 0
                b[k] += (-1.0 * boundary_func(X[i, j], 0))  # u_{i,j-2}
                b[k] += (16.0 * boundary_func(X[i, j], 0))  # u_{i,j-1}
            elif j == Ny - 2:
                # u_{i,j+2} = 0
                b[k] += (-1.0 * boundary_func(X[i, j], 1))  # u_{i,j+2}
                b[k] += (16.0 * boundary_func(X[i, j], 1))  # u_{i,j+1}

    return b
```
##### 3.1.3.4. Solving the Fourth-Order Linear System
```python
def solve_fourth_order_sparse(Nx, Ny, f_func, boundary_func):
    """
    Solves the Poisson equation using a fourth-order sparse direct solver.

    Parameters:
    - Nx, Ny: Number of interior points along x and y axes.
    - f_func: Source term function f(x, y).
    - boundary_func: Boundary condition function g(x, y).

    Returns:
    - u_numeric: 2D array of numerical solution values.
    - solve_time: Time taken to solve.
    """
    # Generate grid
    x, y, X, Y, h_x, h_y = generate_grid(Nx, Ny, Lx=1.0, Ly=1.0)

    # Assemble RHS vector with boundary conditions
    b = assemble_rhs_fourth_order(Nx, Ny, X, Y, f_func, boundary_func, h_x)

    # Assemble Sparse Matrix
    A_sparse = assemble_sparse_matrix_fourth_order(Nx, Ny, h_x)

    # Solve the linear system and time the operation
    start_time = time.time()
    u = spsolve(A_sparse, b)
    end_time = time.time()
    solve_time = end_time - start_time

    # Reshape solution to 2D grid
    u_numeric = u.reshape((Nx, Ny), order='F')  # Column-major

    return u_numeric, solve_time
```
##### 3.1.3.5. Fourth-Order Dense Direct Solver (For Comparison)
```python
def solve_fourth_order_dense(A_dense, b_dense):
    """
    Solves the Poisson equation using a dense fourth-order direct solver.

    Parameters:
    - A_dense: Dense matrix A.
    - b_dense: Dense RHS vector b.

    Returns:
    - u_numeric: 1D array of numerical solution values.
    - solve_time: Time taken to solve.
    """
    start_time = time.time()
    u = np.linalg.solve(A_dense, b_dense)
    end_time = time.time()
    solve_time = end_time - start_time
    return u, solve_time
```
## 3.2. Convergence Study for the Fourth-Order Scheme
### 3.2.1. Implementation of Convergence Study

Objective:

Assess the accuracy and convergence rate of the fourth-order finite difference solver by solving the Poisson equation on grids of varying sizes and computing the relative error in the maximum norm.
```python
def compute_relative_error(u_numeric, u_exact):
    """
    Computes the relative error in the maximum norm between numerical and exact solutions.

    Parameters:
    - u_numeric: 2D array of numerical solution values.
    - u_exact: 2D array of exact solution values.

    Returns:
    - relative_error: Relative error in maximum norm.
    """
    error = np.abs(u_numeric - u_exact)
    relative_error = np.max(error) / np.max(np.abs(u_exact))
    return relative_error

def convergence_study_fourth_order(grid_sizes):
    """
    Performs a convergence study using the fourth-order finite difference solver.

    Parameters:
    - grid_sizes: List of tuples indicating (Nx, Ny) grid sizes.

    Returns:
    - hs: List of step sizes.
    - errors_sparse: List of relative errors for sparse solver.
    - errors_dense: List of relative errors for dense solver.
    - times_sparse: List of computational times for sparse solver.
    - times_dense: List of computational times for dense solver.
    """
    hs = []
    errors_sparse = []
    errors_dense = []
    times_sparse = []
    times_dense = []

    for Nx, Ny in grid_sizes:
        # Assemble and solve using Sparse Direct Solver
        u_sparse, time_sparse = solve_fourth_order_sparse(Nx, Ny, exact_rhs, lambda x, y: 0.0)

        # Generate grid and compute exact solution
        x, y, X, Y, h_x, h_y = generate_grid(Nx, Ny, Lx=1.0, Ly=1.0)
        U_exact = exact_solution(X, Y)

        # Compute Relative Error for Sparse Solver
        rel_error_sparse = compute_relative_error(u_sparse, U_exact)

        # Assemble dense matrix and RHS
        b_fourth_dense = assemble_rhs_fourth_order(Nx, Ny, X, Y, exact_rhs, lambda x, y: 0.0, h_x)
        A_fourth_dense = assemble_sparse_matrix_fourth_order(Nx, Ny, h_x).toarray()

        # Solve using Fourth-Order Dense Direct Solver
        u_dense, time_dense = solve_fourth_order_dense(A_fourth_dense, b_fourth_dense)
        u_dense_reshaped = u_dense.reshape((Nx, Ny), order='F')

        # Compute Relative Error for Dense Solver
        rel_error_dense = compute_relative_error(u_dense_reshaped, U_exact)

        # Record step size
        h = 1.0 / (Nx + 1)
        hs.append(h)
        errors_sparse.append(rel_error_sparse)
        errors_dense.append(rel_error_dense)
        times_sparse.append(time_sparse)
        times_dense.append(time_dense)

        print(f"Grid Size: {Nx}x{Ny}, Step size: {h:.4f}, Sparse Error: {rel_error_sparse:.5e}, Dense Error: {rel_error_dense:.5e}, Sparse Time: {time_sparse:.6f}s, Dense Time: {time_dense:.6f}s")

    return hs, errors_sparse, errors_dense, times_sparse, times_dense

def plot_convergence_fourth_order(hs, errors_sparse, errors_dense):
    """
    Plots the convergence of the fourth-order solver in a log-log scale.

    Parameters:
    - hs: List of step sizes.
    - errors_sparse: List of relative errors for sparse solver.
    - errors_dense: List of relative errors for dense solver.
    """
    plt.figure(figsize=(8,6))
    plt.loglog(hs, errors_sparse, 'o-', label='Sparse Solver')
    plt.loglog(hs, errors_dense, 's-', label='Dense Solver')
    plt.loglog(hs, [errors_sparse[0]*(h/hs[0])**4 for h in hs], 'k--', label='O(h^4) Reference')
    plt.xlabel('Step size (h)')
    plt.ylabel('Relative Error (Maximum Norm)')
    plt.title('Convergence Study: Fourth-Order Finite Difference Solver')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
```
