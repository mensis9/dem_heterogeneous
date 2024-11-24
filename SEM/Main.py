"""
Main script. Run to execute DEM/SEM training and show results.
"""
import os

from NeuralNetwork import *
from Integration import *
from MaterialModel import *
from Plotting import *
from Input import *


# class for DEM/SEM with variable hyperparameters as input
class EnergyMethod:
    def __init__(self, k_dens, k_func, dim_hid, no_layers, act_func, std_dev_rff, optimizer, lr, beta, mb_factor, epochs, directory):
        self.directory = directory
        # instantiate neural network
        if method == 'DEM':
            self.model = FCDNN(dim_hid, no_layers, act_func, std_dev_rff)
        elif method == 'SEM':
            self.model = RBFNN(k_dens, k_func, X)
        # move model to device
        self.model.to(dev)
        # instantiate integration rule
        self.integration = Integration(int_type=int_type)
        # instantiate material models
        self.material1 = MaterialModel(E=E1, nu=nu1)
        if multiphase:
            self.material2 = MaterialModel(E=E2, nu=nu2)
        else:
            self.material2 = None
        # optimizer
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        # maximum number of epochs
        self.epochs = epochs
        if optimizer == 'Adam':
            self.epochs *= 20
        # factor for weighting multiphase boundary loss
        self.beta = beta
        self.mb_factor = mb_factor

    # call model to get displacement and enforce dirichlet BC - needs to be adjusted according to problem!
    def getU(self, X):
        U = self.model(X)
        U[:, 0] *= X[:, 0]
        U[:, 1] *= X[:, 0]
        return U

    # calculate potential energy as loss
    def getLoss(self, losses):
        criterion = nn.MSELoss(reduction='mean')
        # get displacement
        U = self.getU(X)
        # calculate strain energy density and stresses
        if len(losses) < 200:
            Psi1, S1 = self.material1.getInnerEnergy('Hooke', X, U)[:2]
        else:
            Psi1, S1 = self.material1.getInnerEnergy(mat_mod1, X, U)[:2]
        # integrate strain energy density
        PSI = self.integration.internalEnergy(Psi1, shape1, dx=dx, dy=dy)
        # replace inner energy and stresses on inner part with second material in case of multiphase model
        if multiphase:
            PSI -= self.integration.internalEnergy(Psi1[ind2], shape2, dx=dx, dy=dy)
            if len(losses) < 200:
                Psi2, S2 = self.material2.getInnerEnergy('Hooke', X, U)[:2]
            else:
                Psi2, S2 = self.material2.getInnerEnergy(mat_mod2, X, U)[:2]
            PSI += self.integration.internalEnergy(Psi2[ind2], shape2, dx=dx, dy=dy)
        # integrate external work from traction forces
        fext_right = torch.matmul(U[-Ny:, :], torch.tensor(force_right).unsqueeze(1))
        fext_upper = torch.matmul(U[Ny - 1::Ny, :], torch.tensor(force_upper).unsqueeze(1))
        T = self.integration.externalEnergy(fext_right, dx=dy)
        T += self.integration.externalEnergy(fext_upper, dx=dx)
        # loss: potential energy
        energy = PSI - T
        # stresses on boundary should be the same when using multiphase model
        if multiphase:
            # loss on material boundary
            mb_loss = criterion(S1[ind2][indBlr, 0, 0], S2[ind2][indBlr, 0, 0])
            mb_loss += criterion(S1[ind2][indBlr, 0, 1], S2[ind2][indBlr, 0, 1])
            mb_loss += criterion(S1[ind2][indBul, 1, 0], S2[ind2][indBul, 1, 0])
            mb_loss += criterion(S1[ind2][indBul, 1, 1], S2[ind2][indBul, 1, 1])
            loss = energy + self.mb_factor * mb_loss
        else:
            loss = energy
            mb_loss = None
        return loss, energy, mb_loss

    def train(self, trial=None):
        converged = False
        earlystopping = False
        # create directory to save results
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        os.mkdir(self.directory)
        # lists to save losses
        losses = []
        mb_losses = []
        energy_losses = []
        # set model to training mode
        self.model.train()
        self.best_epoch = 1
        torch.save(self.model.state_dict(), self.directory + '/BestState')
        # record time
        start_time = time.time()
        # training loop
        for epoch in range(self.epochs):
            # initialize iteration to save loss in each iteration of L-BFGS
            self.iteration = 0

            # closure function to calculate losses and optimize parameters
            def closure():
                # calculate losses
                loss, energy, mb_loss = self.getLoss(losses)
                if trial is None:
                    if multiphase:
                        print('Epoch %i/%i, Total loss: %.9e, Energy: %.9e, Material interface: %.9e'
                              % (epoch + 1, self.epochs, loss.item(), energy.item(), mb_loss.item()))
                    else:
                        print('Epoch %i/%i, Total loss: %.9e, Energy: %.9e'
                              % (epoch + 1, self.epochs, loss.item(), energy.item()))
                losses.append(loss.detach().cpu())
                energy_losses.append(energy.detach().cpu())
                if multiphase:
                    mb_losses.append(mb_loss.detach().cpu())
                    # save checkpoint of current model
                    if len(losses) > 800 and mb_losses[-1] < np.min(mb_losses[799:-1]):
                        torch.save(self.model.state_dict(), self.directory + '/BestState')
                        self.best_epoch = epoch
                # zero out gradients
                self.optimizer.zero_grad()
                # perform backward pass, i.e. differentiate loss w.r.t. model parameters
                loss.backward(inputs=[parameter for parameter in self.model.parameters()])
                self.iteration += 1
                return loss

            # early stopping criteria
            if len(losses) > 800:
                # early stopping in case of convergence or nan
                rel_loss = np.abs((np.mean(losses[-10:]) - np.mean(losses[-20:-10])) / np.mean(losses[-10:]))
                if rel_loss < conv or np.isnan(losses[-1]):
                    converged = True
                    break
                # report current loss and epoch; prune unpromising trial (only for hyperparameter optimization)
                if trial is not None:
                    trial.report(losses[-1], epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if multiphase:
                    # early stopping if mb_loss starts rising
                    if np.mean(mb_losses[-20:]) > 5 * np.min(mb_losses[400:-1]):
                        self.model.load_state_dict(torch.load(self.directory + '/BestState'))
                        earlystopping = True
                        break
            # perform optimizer step, i.e. update model parameters
            self.optimizer.step(closure)
            # adjust mb_factor every ... epochs
            if multiphase and (optimizer == 'LBFGS' or (optimizer == 'Adam' and (epoch + 1) % 20 == 0)):
                loss, energy_loss, mb_loss = self.getLoss(losses)
                # compute gradients w.r.t. each loss term separately
                self.optimizer.zero_grad()
                energy_loss.backward(retain_graph=True, inputs=[parameter for parameter in self.model.parameters()])
                energy_grads = torch.zeros(0)
                for name, parameter in self.model.named_parameters():
                    energy_grads = torch.cat((energy_grads, parameter.grad.flatten()), 0)
                self.optimizer.zero_grad()
                mb_loss.backward(retain_graph=True, inputs=[parameter for parameter in self.model.parameters()])
                mb_grads = torch.zeros(0)
                for name, parameter in self.model.named_parameters():
                    mb_grads = torch.cat((mb_grads, parameter.grad.flatten()), 0)
                # update rule for material boundary loss
                mb_factor = torch.linalg.norm(energy_grads) / torch.linalg.norm(mb_grads)
                self.mb_factor = self.beta * self.mb_factor + (1 - self.beta) * mb_factor
                if trial is None:
                    print('Updated mb_factor to: %.9e' % mb_factor)
        # compute and print loss on trained model
        loss, energy_loss, mb_loss = self.getLoss(losses)
        if multiphase:
            print('Final loss: %.9e, Energy: %.9e, Material interface: %.9e'
                  % (loss.item(), energy_loss.item(), mb_loss.item()))
        else:
            print('Final loss: %.9e, Energy loss: %.9e'
                  % (loss.item(), energy_loss.item()))
        # print convergence/stop
        if converged:
            print('Converged after %i epochs' % epoch)
        elif earlystopping:
            print('Stopped training early after %i epochs, best loss in epoch %i' % (epoch, self.best_epoch))
            epoch = self.best_epoch
        else:
            epoch += 1
            print('Stopped training after %i epochs' % epoch)
        # save loss on trained model
        losses.append(loss.detach().cpu())
        energy_losses.append(energy_loss.detach().cpu())
        if multiphase:
            mb_losses.append(mb_loss.detach().cpu())
        print('Total training time: %.2f' % (time.time() - start_time))
        return losses, energy_losses, mb_losses, epoch


if __name__ == '__main__':
    print(f"Using {str(dev)} device")
    print('Method: %s, Material model: %s, Integration: %s, Heterogeneous material: %s, Force: %s, Sample points: %i'
          % (method, mat_mod1, int_type, str(multiphase), str(force_right), Nx * Ny))
    print('Saving to %s' % directory)
    if method == 'DEM':
        print('optimizer: %s, lr: %e, dim_hid: %i, no_layers: %i, act_func: %s, std_dev_rff: %e, mb_factor: %e, beta: %e'
              % (optimizer, lr, dim_hid, no_layers, act_func, std_dev_rff, mb_factor, beta))
    elif method == 'SEM':
        print('optimizer: %s, lr: %e, k_dens: %i, k_func: %s, mb_factor: %e, beta: %e'
              % (optimizer, lr, k_dens, k_func, mb_factor, beta))
    # instantiate DEM/SEM class dependant of variable hyperparameters
    energymethod = EnergyMethod(k_dens, k_func, dim_hid, no_layers, act_func, std_dev_rff, optimizer, lr, beta, mb_factor, epochs, directory)
    # plot initial configuration
    plotInitial()
    # train model
    losses, energy_losses, mb_losses, epoch = energymethod.train()
    # evaluate model and detach tensors
    U = energymethod.getU(X)
    S1 = energymethod.material1.getInnerEnergy(mat_mod1, X, U)[-1].cpu().detach()
    S = S1
    if multiphase:
        S2 = energymethod.material2.getInnerEnergy(mat_mod2, X, U)[-1].cpu().detach()
        S[ind2.cpu().detach(), :, :] = S2[ind2.cpu().detach(), :, :]
    else:
        S2 = None
    U = U.cpu().detach()
    # print result
    print(U)
    # save losses
    np.save(directory + '/Losses', losses)
    np.save(directory + '/EnergyLosses', energy_losses)
    np.save(directory + '/MBLosses', mb_losses)
    np.save(directory + '/BestEpoch', epoch)
    # save coordinates
    np.save(directory + '/X', X.cpu().detach())
    # save displacements on whole domain
    np.save(directory + '/U', U)
    # save model
    torch.save(energymethod.model.state_dict(), directory + '/Model')
    # save stresses
    np.save(directory + '/S', S)
