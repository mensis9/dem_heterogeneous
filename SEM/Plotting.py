from Input import *
import matplotlib.pyplot as plt
import numpy as np


# get tensors from device
X = X.cpu().detach()
ind1 = ind1.cpu().detach()
ind2 = ind2.cpu().detach()
indBlr = indBlr.cpu().detach()
indBul = indBul.cpu().detach()


# plot initial configuration with BCs
def plotInitial():
    # inner points
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='black')
    # left boundary
    plt.scatter(X[:Ny, 0], X[:Ny, 1], c='green')
    # right boundary
    plt.scatter(X[-Ny:, 0], X[-Ny:, 1], c='red')
    if multiphase:
        plt.scatter(X[ind2, 0], X[ind2, 1], facecolors='none', edgecolors='orange')
        plt.scatter(X[ind2][indBlr, 0], X[ind2][indBlr, 1], c='magenta')
        plt.scatter(X[ind2][indBul, 0], X[ind2][indBul, 1], c='magenta')
        plt.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.01, 1),
                   labels=['Material 1', 'Dirichlet BC', 'Neumann BC', 'Material 2', 'Material boundary'], markerscale=3)
    else:
        plt.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.01, 1),
                   labels=['Inner points', 'Dirichlet BC', 'Neumann BC'], markerscale=3)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.gca().set_aspect('equal')
    plt.show()


# plot undeformed and deformed configuration
def plotDef(U):
    plt.scatter(X[:, 0], X[:, 1], c='black')
    plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c='blue')
    plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(-0.01, 1.14),
               labels=['Initial configuration', 'Deformed configuration'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.savefig(directory + '/Plots/Deformed')
    plt.show()


# plot stresses
def plotStress(U, S1, S2):
    plt.figure(1)
    # mises stress
    S1mises = torch.sqrt(S1[:, 0, 0]**2 - S1[:, 0, 0] * S1[:, 1, 1] + S1[:, 1, 1]**2 + 3 * S1[:, 1, 0]**2)
    if multiphase:
        S2mises = torch.sqrt(S2[:, 0, 0]**2 - S2[:, 0, 0] * S2[:, 1, 1] + S2[:, 1, 1]**2 + 3 * S2[:, 1, 0]**2)
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1mises,
                    vmin=torch.min(torch.min(S1mises[ind1]), torch.min(S2mises[ind2])),
                    vmax=torch.max(torch.max(S1mises[ind1]), torch.max(S2mises[ind2])))
        plt.scatter(X[ind2, 0] + U[ind2, 0], X[ind2, 1] + U[ind2, 1], c=S2mises[ind2],
                    vmin=torch.min(torch.min(S1mises[ind1]), torch.min(S2mises[ind2])),
                    vmax=torch.max(torch.max(S1mises[ind1]), torch.max(S2mises[ind2])))
    else:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1mises)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='Mises')
    # plt.savefig(directory + '/Plots/Smises')
    # xx stress
    plt.figure(2)
    if multiphase:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 0, 0],
                    vmin=torch.min(torch.min(S1[ind1, 0, 0]), torch.min(S2[ind2, 0, 0])),
                    vmax=torch.max(torch.max(S1[ind1, 0, 0]), torch.max(S2[ind2, 0, 0])))
        plt.scatter(X[ind2, 0] + U[ind2, 0], X[ind2, 1] + U[ind2, 1], c=S2[ind2, 0, 0],
                    vmin=torch.min(torch.min(S1[ind1, 0, 0]), torch.min(S2[ind2, 0, 0])),
                    vmax=torch.max(torch.max(S1[ind1, 0, 0]), torch.max(S2[ind2, 0, 0])))
    else:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 0, 0])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='XX')
    # plt.savefig(directory + '/Plots/Sxx')
    # xy stress
    plt.figure(3)
    if multiphase:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 1, 0],
                    vmin=torch.min(torch.min(S1[ind1, 1, 0]), torch.min(S2[ind2, 1, 0])),
                    vmax=torch.max(torch.max(S1[ind1, 1, 0]), torch.max(S2[ind2, 1, 0])))
        plt.scatter(X[ind2, 0] + U[ind2, 0], X[ind2, 1] + U[ind2, 1], c=S2[ind2, 1, 0],
                    vmin=torch.min(torch.min(S1[ind1, 1, 0]), torch.min(S2[ind2, 1, 0])),
                    vmax=torch.max(torch.max(S1[ind1, 1, 0]), torch.max(S2[ind2, 1, 0])))
    else:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 1, 0])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='YX')
    # plt.savefig(directory + '/Plots/Syx')
    # yy stress
    plt.figure(4)
    if multiphase:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 1, 1],
                    vmin=torch.min(torch.min(S1[ind1, 1, 1]), torch.min(S2[ind2, 1, 1])),
                    vmax=torch.max(torch.max(S1[ind1, 1, 1]), torch.max(S2[ind2, 1, 1])))
        plt.scatter(X[ind2, 0] + U[ind2, 0], X[ind2, 1] + U[ind2, 1], c=S2[ind2, 1, 1],
                    vmin=torch.min(torch.min(S1[ind1, 1, 1]), torch.min(S2[ind2, 1, 1])),
                    vmax=torch.max(torch.max(S1[ind1, 1, 1]), torch.max(S2[ind2, 1, 1])))
    else:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 1, 1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='YY')
    # plt.savefig(directory + '/Plots/Syy')
    # yx stress
    plt.figure(5)
    if multiphase:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 0, 1],
                    vmin=torch.min(torch.min(S1[ind1, 0, 1]), torch.min(S2[ind2, 0, 1])),
                    vmax=torch.max(torch.max(S1[ind1, 0, 1]), torch.max(S2[ind2, 0, 1])))
        plt.scatter(X[ind2, 0] + U[ind2, 0], X[ind2, 1] + U[ind2, 1], c=S2[ind2, 0, 1],
                    vmin=torch.min(torch.min(S1[ind1, 0, 1]), torch.min(S2[ind2, 0, 1])),
                    vmax=torch.max(torch.max(S1[ind1, 0, 1]), torch.max(S2[ind2, 0, 1])))
    else:
        plt.scatter(X[:, 0] + U[:, 0], X[:, 1] + U[:, 1], c=S1[:, 1, 0])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='XY')
    # plt.savefig(directory + '/Plots/Sxy')
    plt.show()
    return


# plot loss curve
def plotLoss(losses, epoch):
    plt.plot(np.linspace(0, epoch, len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig(directory + '/Plots/Loss')
    plt.show()
