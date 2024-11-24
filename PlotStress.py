import matplotlib.pyplot as plt
from SEM.Input import *
import numpy as np
plt.rcParams['figure.figsize'] = (15, 6.5)
plt.rcParams['lines.markersize'] = 2.5

directory_fem = 'FEM/Output/FEMNeoHookeTrueN40Q9Traction/'
x_fem = np.load(directory_fem + 'X.npy')
u_fem = np.load(directory_fem + 'U.npy')
s_fem = np.load(directory_fem + 'S.npy')

directory_dem = 'SEM/' + directory + '/'
x_dem = np.load(directory_dem + 'X.npy')
u_dem = np.load(directory_dem + 'U.npy')
s_dem = np.load(directory_dem + 'S.npy')

plt.figure(1)
plt.scatter(x_dem[:, 0] + u_dem[:, 0], x_dem[:, 1] + u_dem[:, 1], c=s_dem[:, 0, 0])
# plt.scatter(x_fem[:, 0] + u_fem[:, 0], x_fem[:, 1] + u_fem[:, 1], c=s_fem[:, 1, 1])
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.colorbar(label=r'$\sigma$')
plt.tight_layout()
plt.show()
