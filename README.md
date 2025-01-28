# Numerical Analysis Project
## Resolution of a 2D Poisson problem 
### Introduction

This repository contains the implementation and analysis of numerical methods to solve the 2D Poisson problem.  

The project focuses on both theoretical understanding and computational performance, offering insights into direct and iterative solvers.

### Objectives 

**Finite Difference Discretization:** Derive and implement a discrete version of the 2D Poisson equation, validate the solver using a known exact solution, and analyze convergence and computational efficiency.

**Linear System Solvers:** Explore both direct and iterative methods for solving the resulting linear system, comparing computational costs and performance.

**Extensions and Enhancements:** Incorporate additional features such as diffusion terms, higher-order finite difference schemes, and advanced boundary conditions.


## Solving the Poisson Equation Using Finite Difference Method

### 1. Finite Difference Discretization

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

