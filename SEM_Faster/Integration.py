"""
Class for Integration of external and internal Energy. Midpoint integration and composite Simpson's rule are
implemented manually, for trapezoidal integration a built-in torch function is used. For simplicity, numerical
integration is only implemented for rectangular shapes.
"""
import torch


class Integration:
    def __init__(self, int_type):
        self.type = int_type

    def internalEnergy(self, f, shape, dx, dy):
        f = f.reshape(shape)
        if self.type == 'Simpson':
            return self.simpson(self.simpson(f, dx=dx), dx=dy)
        elif self.type == 'Midpoint':
            return self.midpoint(self.midpoint(f, dx=dx), dx=dy)
        elif self.type == 'Trapezoidal':
            return torch.trapezoid(torch.trapezoid(f, dx=dx, dim=1), dx=dy, dim=0)

    def externalEnergy(self, f, dx):
        if self.type == 'Simpson':
            return self.simpson(f, dx=dx)
        elif self.type == 'Midpoint':
            return self.midpoint(f, dx=dx)
        elif self.type == 'Trapezoidal':
            return torch.trapezoid(f, dx=dx, dim=0)

    # numerically integrates f with spacing dx by evenly spaced composite simpsons 1/3 rule
    @staticmethod
    def simpson(f, dx):
        N = f.shape[0]
        if N % 2 == 0:
            # simpsons rule on second to last intervals
            val = 0.5 * dx * (f[-1] + f[-2])
            result = (dx / 3.0 * (f[0:N-3:2] + 4 * f[1:N-2:2] + f[2:N-1:2])).sum(0)
            # simpsons rule on first to next-to-last intervals
            val += 0.5 * dx * (f[0] + f[1])
            result += (dx / 3.0 * (f[1:N-2:2] + 4 * f[2:N-1:2] + f[3:N:2])).sum(0)
            # average
            result = (result + val) / 2
        else:
            result = (dx / 3.0 * (f[0:N-2:2] + 4 * f[1:N-1:2] + f[2:N:2])).sum(0)
        return result

    # numerically integrates f with spacing dx by midpoint rule
    @staticmethod
    def midpoint(f, dx):
        # inner points
        result = dx * f[1:-1].sum(0)
        # corners
        result += 0.5 * dx * f[[0, -1]].sum(0)
        return result
