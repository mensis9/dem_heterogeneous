import numpy as np
import ufl
import os
import shutil
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import matplotlib.pyplot as plt


# structural parameters
Lx = 2
Ly = 1
# number of elements per unit length
N = 40
# lower left and upper right corner of inner material
ll = [0.75, 0.25]
ur = [1.25, 0.75]
# force on right boundary
f_right = (0.5, 0.0)
# element and quadrature degree
element_degree = 2
quadrature_degree = 4

# material parameters
matmod = 'Hooke'
multiphase = True
E1 = default_scalar_type(1.)
if multiphase:
    E2 = default_scalar_type(10.)
else:
    E2 = E1
nu = default_scalar_type(0.3)

# mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Lx * N, Ly * N], mesh.CellType.quadrilateral)
Q = fem.functionspace(domain, ("DG", 0))
V = fem.functionspace(domain, ("CG", element_degree, (domain.geometry.dim, )))


# left end of beam
def left(x):
    return np.isclose(x[0], 0)


# right end of beam
def right(x):
    return np.isclose(x[0], Lx)


# outer material
def domain1(x):
    return np.logical_or(np.logical_or(x[0] <= ll[0], x[0] >= ur[0]), np.logical_or(x[1] <= ll[1], x[1] >= ur[1]))


# inner material
def domain2(x):
    return np.logical_and(np.logical_and(ll[0] <= x[0], x[0] <= ur[0]), np.logical_and(ll[1] <= x[1], x[1] <= ur[1]))


# mark cells for bcs
fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left)
right_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right)
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag_bc = mesh.meshtags(domain, domain.topology.dim - 1, marked_facets[sorted_facets], marked_values[sorted_facets])

# define mu and lambda differently for both materials
cells1 = mesh.locate_entities(domain, domain.topology.dim, domain1)
cells2 = mesh.locate_entities(domain, domain.topology.dim, domain2)
mu = fem.Function(Q)
lam = fem.Function(Q)
mu.x.array[cells1] = np.full_like(cells1, E1 / (2 * (1 + nu)), dtype=default_scalar_type)
mu.x.array[cells2] = np.full_like(cells2, E2 / (2 * (1 + nu)), dtype=default_scalar_type)
lam.x.array[cells1] = np.full_like(cells1, E1 * nu / ((1 + nu) * (1 - 2 * nu)), dtype=default_scalar_type)
lam.x.array[cells2] = np.full_like(cells2, E2 * nu / ((1 + nu) * (1 - 2 * nu)), dtype=default_scalar_type)

# dirchlet bc
u_zero = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)
left = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
dirichlet_bc = [fem.dirichletbc(u_zero, left, V)]

# test and solution functions
v = ufl.TestFunction(V)
u = fem.Function(V)
# traction
t = fem.Constant(domain, default_scalar_type(f_right))

# spatial dimension
d = len(u)
# identity tensor
I = ufl.variable(ufl.Identity(d))
# deformation gradient
F = ufl.variable(I + ufl.grad(u))
# right Cauchy-Green tensor
C = ufl.variable(F.T * F)
# invariants of deformation gradient
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))
# lagrange strain
eps = ufl.sym(ufl.grad(u))

# strain energy
if matmod == 'NeoHooke':
    psi = (mu/2) * (Ic - 2) - mu * ufl.ln(J) + (lam / 2) * (ufl.ln(J)) ** 2
elif matmod == 'Hooke':
    sig = 2 * mu * eps + lam * ufl.tr(eps) * I
    psi = 0.5 * ufl.inner(eps, sig)

# set gauß quadrature degree
metadata = {'quadrature_degree': quadrature_degree}
# measure inner domain
dx = ufl.Measure('dx', domain=domain, metadata=metadata)
# measure right end
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag_bc, metadata=metadata)

# potential energy (integrate strain energy on inner and outer material, traction energy on right boundary)
Pi = psi*dx - ufl.dot(t, u)*ds(2)
# variation of Pi
dPidu = ufl.derivative(Pi, u, v)

# define problem F(u) = 0
problem = NonlinearProblem(dPidu, u, dirichlet_bc)
# set solver and options
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# solve
solver.solve(u)

# get stress
if matmod == 'NeoHooke':
    stress = mu * F + (lam * ufl.ln(ufl.det(F)) - mu) * ufl.inv(F).T
elif matmod == 'Hooke':
    stress = 2 * mu * eps + lam * ufl.tr(eps) * I


# project solution data
W = fem.functionspace(domain, ("DG", 0))
x_unsorted = np.round(np.array(W.tabulate_dof_coordinates()), decimals=4)
nr_elements_x = len(np.unique(x_unsorted[:, 0]))
nr_elements_y = len(np.unique(x_unsorted[:, 1]))
nr_elements = nr_elements_x * nr_elements_y

# all outputs in one array so that they can be sorted
out_unsorted = np.zeros((nr_elements, 8))
# coordinates
out_unsorted[:, :2] = x_unsorted[:, :2]
# displacements
u_expr = fem.Expression(u[0], W.element.interpolation_points())
u_temp = fem.Function(W)
u_temp.interpolate(u_expr)
out_unsorted[:, 2] = u_temp.x.array
u_expr = fem.Expression(u[1], W.element.interpolation_points())
u_temp = fem.Function(W)
u_temp.interpolate(u_expr)
out_unsorted[:, 3] = u_temp.x.array
# stresses
s = np.zeros((nr_elements, 4))
for i in range(2):
    for j in range(2):
        s_expr = fem.Expression(stress[i, j], W.element.interpolation_points())
        s_temp = fem.Function(W)
        s_temp.interpolate(s_expr)
        out_unsorted[:, 4+2*i+j] = s_temp.x.array

# sort output array
out_x_lines = np.split(out_unsorted[np.argsort(out_unsorted[:, 0])], nr_elements_x, axis=0)
out = np.ndarray((nr_elements, 8))
for x in range(nr_elements_x):
    out[nr_elements_y * x:nr_elements_y * x + nr_elements_y] = out_x_lines[x][np.argsort(out_x_lines[x][:, 1])]

# save coordinates, displacements and stresses
directory = ('Output/FEM' + matmod + str(multiphase) + 'N' + str(N) + 'Q' +
             str(4 * element_degree + int(np.floor(0.5 * element_degree))))
if os.path.exists(directory):
    shutil.rmtree(directory)
os.mkdir(directory)
np.save(directory + '/X.npy', out[:, :2])
np.save(directory + '/U.npy', out[:, 2:4])
np.save(directory + '/S.npy', out[:, 4:].reshape(-1, 2, 2))
