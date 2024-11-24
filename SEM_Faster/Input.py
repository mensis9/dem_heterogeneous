"""
Input of structural, material and network parameters as well as specifics of multiphase model and potential analytical
solution.
"""
# ------------------------------ IMPORT --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import optuna
import cProfile
import os
import shutil
import numpy as np

# ------------------------------ PACKAGE SETTINGS ----------------------------------------------------------------------
# torch settings
torch.manual_seed(2024)
torch.set_default_dtype(torch.float32)
# matplotlib settings
plt.rcParams["text.usetex"] = True
plt.rcParams['lines.markersize'] = 3
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Serif'
plt.rcParams['font.size'] = 25
# plt.rcParams['text.latex.preamble'] = r'\boldmath'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['figure.figsize'] = (15, 10)

# ------------------------------ SET DEVICE ----------------------------------------------------------------------------
if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')
torch.set_default_device(dev)

# ------------------------------ STRUCTURAL PARAMETERS -----------------------------------------------------------------
dim_inp = 2  # number of spacial dimensions of mechanical problem (input dimension to neural network)
dim_out = 2  # number of spacial dimensions of mechanical problem (output dimension from neural network)
Lx = 2  # length of beam in x
Ly = 1  # height of beam in y
N = 80
Nx = N * Lx + 1  # number of sample points for network training in x
Ny = N * Ly + 1  # number of sample points for network training in y
shape1 = (Nx, Ny)  # shape of domain tensor
dx = Lx / (Nx - 1)  # distance between sample points in x
dy = Ly / (Ny - 1)  # distance between sample points in y
# sample point coordinates
X = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0, Ly, Ny), indexing='ij')
X = torch.cat((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), dim=1).requires_grad_(True)
# neumann boundary conditions
force_right = [0.5, 0.0]  # prescribed force on right boundary [x, y] (neumann BC)
force_upper = [0.0, 0.0]  # prescribed force on upper boundary [x, y] (neumann BC)

# ------------------------------ NETWORK PARAMETERS --------------------------------------------------------------------
method = 'SEM'  # method to use, i.e. which NN architecture should be applied (SEM/DEM using RBFNN/FCDNN respectively)
dim_hid = 64  # number of neurons in each hidden layer (FCDNN)
no_layers = 7  # number of hidden layers (FCDNN)
act_func = 'Tanh'  # activation function (FCDNN)
rff_mapping = True  # apply random Fourier feature mapping to input (FCDNN)
std_dev_rff = 2.381262e+00  # standard deviation for Gaussian RFF mapping
k_dens = 5  # kernel center density (RFBNN)
k_func = 'matern32'  # kernel function (RBFNN)
optimizer = 'LBFGS'  # optimizer
epochs = 1000  # maximum number of training epochs to perform (LBFGS Adam does 20x the amount)
conv = 1e-5  # training will stop early if: loss[epoch] - loss[epoch-1] < conv
lr = 2.925422e-01  # learning rate
std_dev_weights = 0.1  # standard deviation when initializing weights and biases of linear layer

# ------------------------------ MATERIAL PARAMETERS -------------------------------------------------------------------
int_type = 'Simpson'  # numerical integration type (Midpoint/Simpson/Trapezoidal)
mat_mod1 = 'Hooke'  # material model (Hooke/NeoHooke)
plane = 'strain'  # plane strain or plane stress
E1 = 1  # young's modulus
nu1 = 0.3  # poisson ratio

# ------------------------------ MULTIPHASE MODEL ----------------------------------------------------------------------
multiphase = True
mb_factor = 6.865778e+00  # factor for weighting multiphase boundary MSE loss
beta = 3.961541e-01
mat_mod2 = 'Hooke'
E2 = 10
nu2 = 0.3
ll = [0.75, 0.25]  # lower left corner of rectangle with material 2
ur = [1.25, 0.75]  # upper right corner of rectangle with material 2
# indexes of inner and outer material points
ind2 = torch.all(torch.logical_and(torch.tensor(ll) <= X, X <= torch.tensor(ur)), axis=1)
ind1 = torch.logical_not(ind2)
# shape of tensor with inner material points
shape2 = (X[ind2, 0].unique().size()[0], X[ind2, 1].unique().size()[0])
# indexes of upper and lower boundary points between materials
indBul = torch.logical_or(X[ind2, 1] == X[ind2, 1].min(),
                          X[ind2, 1] == X[ind2, 1].max())
# indexes of left and right boundary points between materials
indBlr = torch.logical_or(X[ind2, 0] == X[ind2, 0].min(),
                          X[ind2, 0] == X[ind2, 0].max())

# file name for saving results
if method == 'DEM':
    if rff_mapping:
        directory = 'Output/' + method + mat_mod1 + str(multiphase) + str(no_layers) + 'x' + str(dim_hid) + 'r' + str(std_dev_rff) + 'N' + str(N)
    else:
        directory = 'Output/' + method + mat_mod1 + str(multiphase) + str(no_layers) + 'x' + str(dim_hid) + 'N' + str(N)
elif method == 'SEM':
    directory = 'Output/' + method + mat_mod1 + str(multiphase) + str(k_dens) + 'N' + str(N)
