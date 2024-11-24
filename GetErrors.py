from SEM.Input import *
from SEM.Integration import *
import numpy as np

integration = Integration(int_type='Simpson')

directory_fem = 'FEM/Output/FEMHookeTrueN40Q9/'
u_fem = np.load(directory_fem + 'U.npy')
s_fem = np.load(directory_fem + 'S.npy')
s_fem_mises = np.sqrt(s_fem[:, 0, 0]**2 - s_fem[:, 0, 0] * s_fem[:, 1, 1] + s_fem[:, 1, 1]**2 + 3 * s_fem[:, 1, 0]**2)

directory_dem = 'SEM/' + directory + '/'
u_dem = np.load(directory_dem + 'U.npy').reshape(Nx, Ny, 2)[1::2, 1::2, :].reshape(-1, 2)
s_dem = np.load(directory_dem + 'S.npy').reshape(Nx, Ny, 2, 2)[1::2, 1::2, :].reshape(-1, 2, 2)
s_dem_mises = np.sqrt(s_dem[:, 0, 0]**2 - s_dem[:, 0, 0] * s_dem[:, 1, 1] + s_dem[:, 1, 1]**2 + 3 * s_dem[:, 1, 0]**2)
loss = np.load(directory_dem + 'Losses.npy')[-1]

shape1 = (N, N // 2)
u = integration.internalEnergy(np.linalg.norm(u_dem - u_fem, axis=1) ** 2, shape1, dx, dy)
s = integration.internalEnergy(np.linalg.norm(np.linalg.norm(s_dem - s_fem, axis=2), axis=1) ** 2, shape1, dx, dy)
u_fem = integration.internalEnergy(np.linalg.norm(u_fem, axis=1) ** 2, shape1, dx, dy)
s_fem = integration.internalEnergy(np.linalg.norm(np.linalg.norm(s_fem, axis=2), axis=1) ** 2, shape1, dx, dy)

error_u = u ** 0.5 / u_fem ** 0.5
error_s = s ** 0.5 / s_fem ** 0.5

print('Loss: %e' % loss[0])
print('Displacement error: %e' % error_u)
print('Stress error: %e' % error_s)
