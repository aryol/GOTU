from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import random
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from utilities import create_test_matrix_11, generate_all_binaries,create_test_matrix_cond, calculate_fourier_coefficients
from examples import tasks
import time

import token_transformer
import models

def build_model(arch, dimension):
    """
    This function creates the model based on the argument given in the command line.
    """
    if arch == 'mlp':
        model = models.MLP(input_dimension=dimension)
    elif arch == 'mup':
        model = models.MaximalUpdate(input_dimension=dimension, width=2**12, depth=6)
    elif arch == 'ntk':
        model = models.NTK(input_dimension=dimension, width=2**13, depth=3)
    elif arch == 'rfrelu':
        model = models.RandomFeaturesReLU(input_dimension=dimension)
    elif arch == 'rfpoly':
        model = models.RandomFeaturesPoly(input_dimension=dimension)
    elif arch == 'meanfield':
        model = models.MeanField(input_dimension=dimension)
    elif arch == 'transformer':
        model = token_transformer.TokenTransformer(
                seq_len=dimension, output_dim=1, dim=256, depth=12, heads=6, mlp_dim=256)
    return model.to(device)



def train(train_X, train_y, valid_X, valid_y, test_X, test_y, computation_interval=0, verbose_interval=0, monomials=None, curr=None, print_coefficients=False):
    """
    This is the main training function which receives the datasets and does the training (curriculum or normal)
    :param monomials: This argument recieves a mask which shows coefficient of which monomials must be computed. 
    :param curr: This argument is used to activate curriculum learning. If none then normal training. If not none, it is equal to (leap, threshold) of the degree-curriculum algorithm.  
    :param computation_interval: Denotes frequency of computation of valid/test losses and also coefficients of the monomials.
    :return: The function returns epoch_logs (just epochs that computations are done), train_losses, valid_losses, test_losses, coefficients (of monomials denoted by monomials argument during the training), coefficients_norms (used for calculating degree profile), iter_counter (number of iterations done during the optimizaiton).
    
    Note that the test dataset is used for the computation of coefficients of the monomials. 
    """
    model = build_model(task_params['model'], dimension)
    # Logging arrays
    epoch_logs = []
    train_losses = []
    valid_losses = []
    test_losses = []
    coefficients = []
    coefficients_norms = []

    # Creating and preparing the frozen dataset
    ## Reshaping
    train_y = train_y.reshape(-1, 1)
    valid_y = valid_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    ## Creating pytorch tensors
    train_X = torch.tensor(train_X, device=device)
    train_y = torch.tensor(train_y, device=device)
    valid_X = torch.tensor(valid_X, device=device)
    valid_y = torch.tensor(valid_y, device=device)
    test_X = torch.tensor(test_X, device=device)
    test_y = torch.tensor(test_y, device=device)

    # Creating datasets and dataloaders
    if curr is None:
        train_ds = TensorDataset(train_X, train_y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_ds = TensorDataset(valid_X, valid_y)
        valid_dl = DataLoader(valid_ds, batch_size=test_batch_size)
    else:
        # Creating the dataset based on the degree curriculum algorithm
        leap_start = 0
        leap_step = curr[0] # curr[0] is the leap of the Hamming balls in the curriculum algorithm. 
        while leap_start == 0 or len(train_dl) == 0:
            leap_start += leap_step
            indices = train_X.sum(dim=1) >= dimension - 2 * leap_start
            train_ds = TensorDataset(train_X[indices], train_y[indices])
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            indices = valid_X.sum(dim=1) >= dimension - 2 * leap_start
            valid_ds = TensorDataset(valid_X[indices], valid_y[indices])
            valid_dl = DataLoader(valid_ds, batch_size=test_batch_size)

    
    test_ds = TensorDataset(test_X, test_y)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size)
    # Defining the optimizer
    if task_params['opt'].lower() == 'sgd':
        opt = optim.SGD(model.parameters(), lr=task_params['lr'], momentum=momentum, weight_decay=0.0)
    else:
        print("Using Adam")
        opt = optim.Adam(model.parameters(), lr=task_params['lr'], weight_decay=0.001)

    loss_func = nn.MSELoss()
    


    # Function used for evaluation of the model, i.e., calculation of coefficients and valid/test losses. 
    def model_evaluation(epoch, train_loss):
        model.eval()
        with torch.no_grad():
            # Computing coefficients of the monomials and the average norm per degree
            if monomials is not None:
                y_pred = torch.vstack([model(xb) for xb, _ in test_dl])
                coefficients.append(calculate_fourier_coefficients(monomials, test_X.cpu().detach().numpy(),
                                                               y_pred.cpu().detach().numpy()))
                coefficients_norms.append([((coefficients[-1][monomials.sum(axis=1) == dim]) ** 2).sum() for dim in range(dimension + 1)])                           
            # Computing loss on the validation and test data
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) # Validation data is defined un the seen domain. 
            if monomials is not None:
                test_loss = loss_func(y_pred, test_y)
            else:
                test_loss = sum(loss_func(model(xb), yb) for xb, yb in test_dl)  # Here the loss is on the whole space omega (seen and unseen). 
                test_loss /= len(test_dl)
            valid_loss /= len(valid_dl)
            if train_loss is None:
                train_loss = valid_loss
            train_loss = train_loss.cpu().detach().numpy()
            valid_loss = valid_loss.cpu().detach().numpy()
            test_loss = test_loss.cpu().detach().numpy()
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            epoch_logs.append(epoch)
            if (epoch >= 0 and epoch % verbose_interval == 0) or epoch == task_params['epochs'] - 1:
                if (monomials is not None) and print_coefficients:
                    print("Coefficient norms:", coefficients_norms[-1])
                    print("Coefficients:", coefficients[-1])
                print(f"Epoch: {epoch:4}, Train Loss: {train_loss:0.6}, Valid Loss: {valid_loss:0.6}, Test Loss: {test_loss:0.6}, Elapsed Time:", time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))

    
    # Model's evaluation before training
    model_evaluation(-1, None)
    iter_counter = 0
    for epoch in range(task_params['epochs']):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        for xb, yb in train_dl:
            iter_counter += 1
            pred = model(xb)
            loss = loss_func(pred, yb)
            train_loss += loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        train_loss /= len(train_dl)
        if (epoch >= 0 and epoch % computation_interval == 0) or epoch == task_params['epochs'] - 1:
            model_evaluation(epoch, train_loss)
        # Updating the curriculum if needed. 
        if (curr is not None) and (train_loss < curr[1]):
            if not leap_start < dimension:
                break
            else:
                leap_start += leap_step
                print(f"Epoch: {epoch} -- curriculum changed to {leap_start}")
                
                indices = train_X.sum(dim=1) >= dimension - 2 * leap_start
                train_ds = TensorDataset(train_X[indices], train_y[indices])
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                indices = valid_X.sum(dim=1) >= dimension - 2 * leap_start
                valid_ds = TensorDataset(valid_X[indices], valid_y[indices])
                valid_dl = DataLoader(valid_ds, batch_size=test_batch_size)
    print("Number of iterations:", iter_counter)
    return epoch_logs, train_losses, valid_losses, test_losses, coefficients, coefficients_norms, iter_counter



if __name__ == '__main__':
    
    parser = ArgumentParser(description="Training script for neural networks on different functions",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Required runtime params
    parser.add_argument('-task', required=True, type=str, help='name of the task')
    parser.add_argument('-model', required=True, type=str, help='name of the model')
    parser.add_argument('-epochs', required=True, type=int, help='number of epochs')
    parser.add_argument('-lr', required=True, type=float, help='learning rate')
    parser.add_argument('-seed', required=True, type=int, help='random seed')
    # Other runtime params
    parser.add_argument('-opt', default='sgd', type=str, help='sgd or adam')
    parser.add_argument('-batch-size', default=64, type=int, help='batch size')
    parser.add_argument('-curr-step', default=0, type=int, help='curriculum step size')
    parser.add_argument('-test-batch-size', type=int, default=8192, help='batch size for test samples')
    parser.add_argument('-verbose-int', default=1, type=int, help="the interval between prints")
    parser.add_argument('-compute-int', default=1, type=int, help="the interval between computations of monomials and losses")
    
    args = parser.parse_args()
    start_time = time.time()
    # General setup of the experiments
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    momentum = 0.0

    if args.task not in tasks:
        print("Task not found.")
        exit()
    task_params = tasks[args.task]
    dimension = task_params['dimension']
    task_params.update(vars(args))
    batch_size = task_params['batch_size']
    test_batch_size = task_params['test_batch_size']
    mask = task_params['mask']
    if task_params['curr_step'] == 0:
        curriculum = None
    else:
        curriculum = [task_params['curr_step'], 0.001]
    if mask.shape[1] < dimension:
        mask = np.hstack((mask, np.zeros((mask.shape[0], dimension - mask.shape[1]), dtype=int)))

    print(vars(args))

    # Setting the seeds
    np.random.seed(task_params['seed'])
    random.seed(task_params['seed'])
    torch.manual_seed(task_params['seed'])

    # Generating train, valid, and test data. We use num_samples = 0 as an indication to create the whole space. 
    if task_params['train_size'] == 0:
        train_X = generate_all_binaries(dimension)
        train_X = train_X[np.random.permutation(train_X.shape[0])]
        train_X = train_X[task_params['seen_condition'](train_X)]
    else:
        train_X = create_test_matrix_cond(task_params['train_size'], dimension, task_params['seen_condition'])
    train_y = task_params['target_function'](train_X)

    if task_params['valid_size'] == 0:
        valid_X = generate_all_binaries(dimension)
        valid_X = valid_X[task_params['seen_condition'](valid_X)]
    else:
        valid_X = create_test_matrix_cond(task_params['valid_size'], dimension, task_params['seen_condition'])
    valid_y = task_params['target_function'](valid_X)

    if task_params['test_size'] == 0:
        test_X = generate_all_binaries(dimension)
    else:
        test_X = create_test_matrix_11(task_params['test_size'], dimension)
    test_y = task_params['target_function'](test_X)

    # Checking the samples
    print(f"Shape of train samples: {train_X.shape}, valid samples: {valid_X.shape}, test samples: {test_X.shape}")

    # Running and saving the results
    epoch_logs, train_losses, valid_losses, test_losses, coefficients, coefficients_norms, iter_counter = train(train_X, train_y, valid_X, valid_y, test_X, test_y, computation_interval=task_params['compute_int'], verbose_interval=task_params['verbose_int'], monomials=mask, curr=curriculum, print_coefficients=task_params['print_coefficients'])
    saved_data = {'epochs': np.array(epoch_logs), 'train_losses': train_losses, 
                  'valid_losses': valid_losses, 'test_losses': test_losses, 'coefficients': coefficients, 'coefficients_norms': coefficients_norms, 
                  'run_params': vars(args), 'curriculum': curriculum, 'iters': iter_counter}
    
    with open(f"{args.task}_{task_params['model']}_{task_params['seed']}_{task_params['lr']}_{task_params['epochs']}_{task_params['opt']}_{task_params['curr_step']}.npz", "wb") as f:
        np.savez(f, **saved_data)
