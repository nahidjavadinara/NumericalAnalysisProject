# NumericalAnalysisProject
## Resolution of a 2D Poisson problem 
### Introduction

This repository contains the implementation and analysis of numerical methods to solve the 2D Poisson problem.  

The project focuses on both theoretical understanding and computational performance, offering insights into direct and iterative solvers.

### Objectives 

**Finite Difference Discretization:** Derive and implement a discrete version of the 2D Poisson equation, validate the solver using a known exact solution, and analyze convergence and computational efficiency.

**Linear System Solvers:** Explore both direct and iterative methods for solving the resulting linear system, comparing computational costs and performance.

**Extensions and Enhancements:** Incorporate additional features such as diffusion terms, higher-order finite difference schemes, and advanced boundary conditions.



***Import Necessary Libraries***
```bash
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
