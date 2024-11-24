"""
Classes for Neural Networks. Radial Basis Function Neural Network is used for Shallow Energy Method, Fully Connected
Deep Neural Network is used for Deep Energy Method.
"""
import torch

from Input import *
import rff
import torch.nn as nn


# radial basis function (rbf) neural network consisting of one rbf-layer followed by one linear layer
class RBFNN(nn.Module):
    def __init__(self, k_dens, k_func, X):
        super(RBFNN, self).__init__()
        # total number of kernel centres
        self.no_centres = (Lx * k_dens + 1) * (Ly * k_dens + 1)
        # RBFNN: rbf layer followed by linear layer
        self.rbf_layer = RBFLayer(k_dens, k_func, X)
        self.linear_layer = nn.Linear(self.no_centres, dim_out)

    def forward(self, x):
        out = self.rbf_layer(x)
        out = self.linear_layer(out)
        return out

    def getDerivative(self):
        return self.rbf_layer.getDerivative()

# rbf layer class
class RBFLayer(nn.Module):
    def __init__(self, k_dens, k_func, X):
        super(RBFLayer, self).__init__()
        self.k_x = int(Lx * k_dens + 1)  # number of kernel centers in x
        self.k_y = int(Ly * k_dens + 1)  # number of kernel centers in y
        self.no_centres = self.k_x * self.k_y  # number of kernel centres
        self.k_func = getattr(RBFLayer, k_func)  # kernel function
        self.X = X
        with torch.no_grad():
            self.centres = torch.zeros(self.no_centres, dim_inp)  # kernel center positions
            self.widths = torch.zeros(self.no_centres)  # kernel widths
            self.setParameters()  # set RBF layer parameters
        self.out = self.getOutput()  # output of rbf-layer, same in every epoch

    # set RBF layer parameters (not learnable)
    def setParameters(self):
        # define kernel center positions (uniformly distributed)
        self.centres[:, 0] = torch.linspace(0, Lx, self.k_x).repeat_interleave(self.k_y)
        self.centres[:, 1] = torch.linspace(0, Ly, self.k_y).repeat(self.k_x)
        # distances between all kernel centres
        centre_distances = torch.cdist(self.centres, self.centres, compute_mode='use_mm_for_euclid_dist', p=2)
        # widths associated with centres
        self.widths = torch.max(centre_distances, dim=0)[0] / (2 * min(self.k_x, self.k_y)) ** 0.5

    # get output of RBF layer (same in every epoch)
    def getOutput(self):
        # distances between kernel centres and point coordinates
        size = (self.X.size(0), self.no_centres, dim_inp)
        distances = torch.linalg.norm(self.X.unsqueeze(1).expand(size) - self.centres.unsqueeze(0).expand(size), dim=2)
        # output with kernel function applied
        out = self.k_func(distances / self.widths.unsqueeze(0))
        return out

    def forward(self, x):
        return self.out

    # get derivative of rbf layer w.r.t. input
    def getDerivative(self):
        dXdRBF = torch.zeros(X.size(0), self.no_centres, 2)
        for i in range(self.no_centres):
            dXdRBF[:, i, :] = torch.autograd.grad(self.out[:, i].unsqueeze(1), X, torch.ones(X.size(0), 1), create_graph=True)[0]
        return dXdRBF.detach()

    @staticmethod
    def matern32(x):
        phi = (torch.ones_like(x) + 3 ** 0.5 * x) * torch.exp(-3 ** 0.5 * x)
        return phi

    @staticmethod
    def matern52(x):
        phi = (torch.ones_like(x) + 5 ** 0.5 * x + (5 / 3) * x.pow(2)) * torch.exp(-5 ** 0.5 * x)
        return phi

    @staticmethod
    def gaussian(x):
        phi = torch.exp(-1 * x.pow(2))
        return phi


# fully connected deep neural network
class FCDNN(nn.Module):
    def __init__(self, dim_hid, no_layers, act_func, std_dev_rff):
        super(FCDNN, self).__init__()
        if rff_mapping:
            # RFF mapping - acts as input layer
            self.encoding = rff.layers.GaussianEncoding(sigma=std_dev_rff, input_size=dim_inp,
                                                        encoded_size=dim_hid // 2)
        else:
            # normal input layer
            self.fc_inp = nn.Linear(dim_inp, dim_hid, bias=True)
            torch.nn.init.constant_(self.fc_inp.bias, 0.)
            torch.nn.init.xavier_normal_(self.fc_inp.weight)
        # hidden layers
        self.fc_hid = nn.ModuleList()
        for i in range(no_layers):
            self.fc_hid.append(nn.Linear(dim_hid, dim_hid, bias=True))
            torch.nn.init.constant_(self.fc_hid[i].bias, 0.)
            torch.nn.init.xavier_normal_(self.fc_hid[i].weight)
        # output layers
        self.fc_out = nn.Linear(dim_hid, dim_out, bias=True)
        torch.nn.init.constant_(self.fc_out.bias, 0.)
        torch.nn.init.xavier_normal_(self.fc_out.weight)
        # activation function
        self.act_func = getattr(torch.nn, act_func)()

    # forward pass - execute linear layer sequence
    def forward(self, x):
        # input layer
        if rff_mapping:
            out = self.encoding(x)
        else:
            out = self.fc_inp(x)
            out = self.act_func(out)
        # hidden layers
        for fc in self.fc_hid:
            out = fc(out)
            out = self.act_func(out)
        # output layer
        out = self.fc_out(out)
        return out
