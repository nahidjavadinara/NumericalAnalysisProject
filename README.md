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
Nx, Ny = 50, 50  # Number of grid points
hx, hy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
```
***Generate grid***
```python
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
```
***Define the source term f(x, y)***
```python
f = np.sin(np.pi * X) * np.sin(np.pi * Y)
```
***Flatten the source term into a 1D vector***
```python
f_flat = f[1:-1, 1:-1].flatten()
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
***Apply boundary conditions (homogeneous Dirichlet)***

Add adjustments to the right-hand side for boundary values if needed

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

### Finding the Right-Hand Side \(f(x, y)\)

Taking the Laplacian of \(u_{\text{ex}}(x, y)\):

$$
\Delta u_{\text{ex}} = \frac{\partial^2 u_{\text{ex}}}{\partial x^2} + \frac{\partial^2 u_{\text{ex}}}{\partial y^2}.
$$

Compute the partial derivatives:

1. First derivative with respect to \(x\):
   $$
   \frac{\partial u_{\text{ex}}}{\partial x} = 2\sin(\pi x)\cos(\pi x)\sin^2(\pi y).
   $$

2. Second derivative with respect to \(x\):
   $$
   \frac{\partial^2 u_{\text{ex}}}{\partial x^2} = 2\pi^2\cos(2\pi x)\sin^2(\pi y).
   $$

3. Similarly, for \(y\):
   $$
   \frac{\partial^2 u_{\text{ex}}}{\partial y^2} = 2\pi^2\cos(2\pi y)\sin^2(\pi x).
   $$

Substituting into \(\Delta u_{\text{ex}}\) gives:

$$
f(x, y) = -\Delta u_{\text{ex}} = -2\pi^2\left(\cos(2\pi x)\sin^2(\pi y) + \cos(2\pi y)\sin^2(\pi x)\right).
$$

### Consistency with Boundary Conditions

The solution \(u_{\text{ex}}(x, y)\) satisfies the homogeneous Dirichlet boundary conditions:

$$
u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0.
$$

### Numerical Validation

#### Relative Error and Convergence

Using a grid with \(N \times N\) points, solve the linear system for the numerical solution \(u_{\text{num}}\) and compute the relative error in the maximum norm:

$$
\text{Error} = \|u_{\text{num}} - u_{\text{ex}}\|_\infty.
$$

#### Convergence Plot

Generate a log-log plot of the error against the step size \(h = \frac{1}{N}\). The expected convergence rate for the second-order finite difference scheme is \(O(h^2)\).

```python
import numpy as np
import matplotlib.pyplot as plt

# Define step sizes and errors
step_sizes = [1/N for N in [10, 20, 40, 80]]
errors = [0.01, 0.0025, 0.000625, 0.00015625]  # Replace with computed errors

# Plot
plt.figure()
plt.loglog(step_sizes, errors, marker='o', label='Numerical Error')
plt.title('Convergence of Finite Difference Method')
plt.xlabel('Step size (h)')
plt.ylabel('Error (log scale)')
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.show()

