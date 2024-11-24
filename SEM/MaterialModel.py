"""
Class for material model. Inner energy and stresses are calculated according to Hooke's law or neo-Hookean material
model.
"""
from Input import *


class MaterialModel:
    def __init__(self, E, nu):
        # stiffness tensor - for elasticity
        self.C = torch.zeros(2, 2, 2, 2)
        if plane == 'strain':
            self.C[[0, 1], [0, 1], [0, 1], [0, 1]] = 1 - nu
            self.C[[0, 1], [0, 1], [1, 0], [1, 0]] = nu
            self.C[[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]] = (1 - 2 * nu) / 2
            self.C *= E / ((1 + nu) * (1 - 2 * nu))
        elif plane == 'stress':
            self.C[[0, 1], [0, 1], [0, 1], [0, 1]] = 1
            self.C[[0, 1], [0, 1], [1, 0], [1, 0]] = nu
            self.C[[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0]] = (1 - nu) / 2
            self.C *= E / (1 - nu ** 2)
        # lame parameters - for hyperelasticity
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))

    # calculate inner energy psi and stresses sigma from coordinates and displacements according to material model
    def getInnerEnergy(self, model, duxdX, duydX):
        # calculate energy depending on material model used
        if model == 'Hooke':
            return self.hooke(duxdX, duydX)
        elif model == 'NeoHooke':
            return self.neohooke(duxdX, duydX)

    # hooke's law (linear elasticity)
    def hooke(self, X, U):
        duxdX = torch.autograd.grad(U[:, 0].unsqueeze(1), X, torch.ones(X.size(0), 1), create_graph=True)[0]
        duydX = torch.autograd.grad(U[:, 1].unsqueeze(1), X, torch.ones(X.size(0), 1), create_graph=True)[0]
        # strains
        eps = torch.zeros(X.size()[0], 2, 2)
        eps[:, 0, 0] = duxdX[:, 0]
        eps[:, 1, 1] = duydX[:, 1]
        eps[:, 0, 1] = 0.5 * (duxdX[:, 1] + duydX[:, 0])
        eps[:, 1, 0] = eps[:, 0, 1]
        # stresses
        sig = torch.einsum('ijk,mnjk->imn', eps, self.C)
        # strain energy density
        psi = 0.5 * torch.einsum('ijk,ijk->i', eps, sig)
        return psi, sig

    # neo-hooke (hyperelasticity)
    def neohooke(self, X, U):
        duxdX = torch.autograd.grad(U[:, 0].unsqueeze(1), X, torch.ones(X.size(0), 1), create_graph=True)[0]
        duydX = torch.autograd.grad(U[:, 1].unsqueeze(1), X, torch.ones(X.size(0), 1), create_graph=True)[0]
        # deformation gradient
        F = torch.zeros(X.size()[0], X.size()[1], X.size()[1])
        F[:, 0, 0] = duxdX[:, 0] + 1
        F[:, 0, 1] = duxdX[:, 1]
        F[:, 1, 0] = duydX[:, 0]
        F[:, 1, 1] = duydX[:, 1] + 1
        # right cauchy green deformation tensor
        C = torch.matmul(torch.transpose(F, dim0=1, dim1=2), F)
        # jacobi-determinant of deformation map
        J = torch.linalg.det(F)
        if plane == 'strain':
            # trace of C (1st invariant)
            trC = torch.sum(torch.diagonal(C, dim1=1, dim2=2), dim=1)
            # strain energy density
            psi = 0.5 * self.lam * torch.pow(torch.log(J), 2) - self.mu * torch.log(J) + 0.5 * self.mu * (trC - 2)
        elif plane == 'stress':
            # right cauchy green deformation tensor with 3d strain component
            C3d = torch.zeros(Nx*Ny, 3, 3)
            C3d[:, :2, :2] = C
            C3d[:, 2, 2] = 1 / J
            # trace of C (1st invariant)
            trC = torch.sum(torch.diagonal(C3d, dim1=1, dim2=2), dim=1)
            # jacobi-determinant of deformation map
            J = torch.sqrt(torch.linalg.det(C3d))
            # strain energy density
            psi = (0.5 * self.lam * torch.pow(torch.log(J), 2) - self.mu * torch.log(J) + 0.5 * self.mu * (trC - 3))
        # 1st piola-kirchhoff stress
        p = torch.autograd.grad(psi, F, torch.ones((psi.size()[0]), device=dev), create_graph=True)[0]
        # cauchy stress
        sig = torch.mul(torch.pow(J.unsqueeze(1).unsqueeze(2), -1), torch.matmul(p, torch.transpose(F, dim0=-2, dim1=-1)))
        return psi, p, sig
