"""''
Hyperparameter optimization, uses Optuna library with TPE sampler.
'"""

from Main import *
import Input as inp
import optuna
import typing


def save(study, trial):
    if len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])) >= 1:
        if study.best_trial == trial:
            torch.save(trial, './Output/Optuna_BestTrial_' + method + str(multiphase))


# objective to be minimized
def objective(trial):
    # directory for saving intermediate results
    directory = 'Output/HPOpt_' + method + mat_mod1 + str(multiphase)
    # variable hyperparameters and respective ranges
    # optimizer = trial.suggest_categorical('optimizer', ['Adam', 'LBFGS'])  # optimizer
    optimizer = inp.optimizer
    # lr = 0.1
    if optimizer == 'Adam':
        lr = trial.suggest_float('lr', 1e-4, 1e-2)  # learning rate
    elif optimizer == 'LBFGS':
        lr = trial.suggest_float('lr', 1e-2, 5e-1)  # learning rate
        # lr = inp.lr
    # std_dev_weights = trial.suggest_float('std_dev_weights', 0, 1)  # standard deviation of initial linear weights
    std_dev_weights = inp.std_dev_weights
    if multiphase == True:
        mb_factor = trial.suggest_float('mb_factor', 0.1, 10)
        # mb_factor = inp.mb_factor
        beta = trial.suggest_float('beta', 0.1, 0.9)
        # beta = inp.beta
    else:
        mb_factor = inp.mb_factor
        beta = inp.beta
    if method == 'DEM':
        dim_hid = trial.suggest_int('dim_hid', 48, 64, step=2)
        # dim_hid = inp.dim_hid
        no_layers = trial.suggest_int('no_layers', 6, 9)
        # no_layers = inp.no_layers
        # act_func = trial.suggest_categorical('act_func', ['Tanh', 'Sigmoid', 'Softsign'])
        act_func = inp.act_func
        std_dev_rff = trial.suggest_float('std_dev_rff', 0.1, 10)
        # std_dev_rff = inp.std_dev_rff
        k_dens, k_func = inp.k_dens, inp.k_func
    elif method == 'SEM':
        dim_hid, no_layers, act_func, std_dev_rff = inp.dim_hid, inp.no_layers, inp.act_func, inp.std_dev_rff
        # k_dens = trial.suggest_int('k_dens', 30, 50)
        k_dens = inp.k_dens
        # k_func = trial.suggest_categorical('k_func', ['gaussian', 'matern32', 'matern52'])
        k_func = inp.k_func
    # instantiate energy method class with variable hyperparameters
    energymethod = EnergyMethod(k_dens, k_func, dim_hid, no_layers, act_func, std_dev_rff, optimizer, lr, beta, mb_factor, epochs, directory)
    # print current hyperparameters
    if method == 'SEM':
        print('optimizer: %s, lr: %e, k_dens: %i, k_func: %s, mb_factor: %e, beta: %e'
              % (optimizer, lr, k_dens, k_func, mb_factor, beta))
    elif method == 'DEM':
        print('optimizer: %s, lr: %e, dim_hid: %i, no_layers: %i, act_func: %s, mb_factor: %e, beta: %e, std_dev_rff: %e'
              % (optimizer, lr, dim_hid, no_layers, act_func, mb_factor, beta, std_dev_rff))
    # train model
    losses, energy_losses, mb_losses, epoch = energymethod.train(trial)
    loss = losses[-1]
    iterations = len(losses)
    # report loss and no. epochs needed for convergence
    trial.report(loss, iterations)
    return loss


# class for multiple pruning methods (https://github.com/optuna/optuna/issues/2042)
class MultiplePruners(optuna.pruners.BasePruner):
    def __init__(self, pruners: typing.Iterable[optuna.pruners.BasePruner]) -> None:
        self.pruners = tuple(pruners)
        self.pruning_condition_check_fn = any

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return self.pruning_condition_check_fn(pruner.prune(study, trial) for pruner in self.pruners)


# create study to find hyperparameters which yield minimum loss value
# median: prune trials that are worse than the median before
# threshold: prune trials that are below the expected value (only possible if estimate is known) or nan
study = optuna.create_study(load_if_exists=True, study_name='hp_optim', direction='minimize',
                            pruner=optuna.pruners.MedianPruner())
                            # pruner=MultiplePruners((optuna.pruners.MedianPruner())))
# optimize study
print(f"Using {str(dev)} device")
print('Method: %s, Material model: %s, Integration: %s, Heterogeneous material: %s, Force: %s, Sample points: %i'
      % (method, mat_mod1, int_type, str(multiphase), str(force_right), Nx * Ny))
study.enqueue_trial({'lr': inp.lr, 'mb_factor': inp.mb_factor, 'beta': inp.beta, 'dim_hid': inp.dim_hid,
                     'no_layers': inp.no_layers, 'std_dev_rff': inp.std_dev_rff, 'k_dens': inp.k_dens})
study.optimize(objective, n_trials=100, callbacks=[save])

# get and print results
pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
print('Study statistics: ')
print('  Number of finished trials: ', len(study.trials))
print('  Number of pruned trials: ', len(pruned_trials))
print('  Number of complete trials: ', len(complete_trials))

# save and print best trial
best = study.best_trial
torch.save(best, 'Output/Optuna/BestTrial_' + method + str(multiphase))
print('Best trial:')
print('  Value: ', best.value)
print('  Params: ')
for key, value in best.params.items():
    print('    {}: {}'.format(key, value))
