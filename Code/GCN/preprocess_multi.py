#Install necessary packages
import numpy as np 
import xarray as xr 
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def preprocess_variable(data_path, variable, window_length, spin_up=None, forecast_horizon = 4):
    """
    Preprocesses one or several variables for use as input and target in the GCN models. 
    All variables are normalized separately and should be entered as a list. 

    Arguments:
    arg[1] : data_path (str) - Full path to train.nc file
    arg[2] : variable (str or list) - Variable names to preprocess - please choose between 'psi', 'q', 'q_adevective'
    arg[3] : window_length (int) - Sliding window size - optional with 4 as default
    arg[4] : spin_up (int) - Number of initial timesteps to skip. If None - all timesteps are included in training

    Returns:
    X_train, y_train, X_test, y_test, X_val, y_val, metadata
    """
    # Allows the user of the code to hand in either a string or a list of variables 
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = list(variable)

    #Opens the train.nc dataset
    data = xr.open_dataset(data_path)

    #Select out run and create train, test & validation, 70%, 15% and 15% accordingly 
    runs = data['run']
    run_train = runs[0:210]
    run_test  = runs[210:255]
    run_val   = runs[255:]

    #Ensure right shapes and that the values are in correct order as we dont want to shuffle the actual runs 
    print(f'Train runs: {run_train.shape}, Test runs: {run_test.shape}, Val runs: {run_val.shape}')
    check = np.concatenate([run_train, run_test, run_val])
    if not np.array_equal(check, runs):
        raise ValueError('Data is not in chronological order 0–299. Please make sure the coordinate run is not shuffled.')

    x_tensors_train, x_tensors_test, x_tensors_val = [], [], []
    #Number of y tensors depend on the forecast horizon. Forecasting horizon = 4 yields four empty lists for exmample 
    y_tensors_train, y_tensors_test, y_tensors_val = [[] for _ in range(forecast_horizon)], [[] for _ in range(forecast_horizon)], [[] for _ in range(forecast_horizon)]
    metadata_vars = {}

    for var_name in variables:
        print(f'\n── Processing variable: {var_name} ──')
        #select given variable
        var = data[var_name]
        #select given time slice if spin up is not None
        if spin_up is not None:
            var = var.isel(time=slice(spin_up, None))

        var_train = var.sel(run=run_train)
        var_test = var.sel(run=run_test)
        var_val  = var.sel(run=run_val)
        #Ensure correct shape of the variable after splitting the runs 
        print(f'The training shape of {var_name}: {var_train.shape}')
        print(f'The test shape of {var_name}: {var_test.shape}')
        print(f'The validation shape of {var_name}: {var_val.shape}')

        #calculate mean and std from the training dataset
        var_mean = var_train.mean(dim=('run', 'time', 'y', 'x'))
        var_std  = var_train.std(dim=('run', 'time', 'y', 'x'))

        #standardizing the data
        def norm(v):
            return (v - var_mean) / var_std

        var_norm_train = norm(var_train)
        var_norm_test  = norm(var_test)
        var_norm_val   = norm(var_val)

        #Creating sliding windows using xarray functions for x 
        def sliding_window(x, window):
            func = x.rolling(time=window).construct('window')
            func = func.isel(time=slice(window - 1, None))
            return func

        x_train_w = sliding_window(var_norm_train, window_length)
        x_test_w  = sliding_window(var_norm_test, window_length)
        x_val_w   = sliding_window(var_norm_val, window_length)
        #Shape after sliding the window is (run, time, lev, y, x, window_length) - 
        #and is reshaped later per PyTorch specifications

        #Creating our target variables using the sliding window technique 
        #input 0,1,2,3 yields target 4,5,6,7 with a forecast horizon of 4 for example
        def get_y_steps(var_norm, split = 'train'):
            T = var_norm.shape[1] #Available time steps in datset after spin up is selected or None
            n_valid = T - window_length - forecast_horizon + 1 #The valid number of windows in dataset 
            y_steps = []
            for h in range(1, forecast_horizon +1):
                start = window_length + h - 1
                end = window_length + h -1 + n_valid
                y_h = var_norm.isel(time=slice(start, end))  
                y_steps.append(y_h)
            return y_steps, n_valid

        y_train_steps, n_train = get_y_steps(var_norm_train)
        y_test_steps, n_test = get_y_steps(var_norm_test)
        y_val_steps, n_val = get_y_steps(var_norm_val)

        #Because the shape of x and y will mismatch by one value after the sliding window, we will cut out one time value from x shapes
        x_train_cut = x_train_w.isel(time=slice(None, n_train))
        x_test_cut  = x_test_w.isel(time=slice(None, n_test))
        x_val_cut   = x_val_w.isel(time=slice(None, n_val))

        #Now we reshape our arrays to adapt them to the PyTorch input shapes 
        def reshape_x(x):
            run, time, lev, Y, X, wl = x.shape
            x_tmp = x.values.reshape(run * time, lev, Y, X, wl) # (population = run * time)
            x_tmp = np.transpose(x_tmp, (0, 2, 3, 1, 4)) # (population, Y, X, lev, window length)
            return x_tmp.reshape(run * time, Y * X, lev * wl) #channels = lev * window_length

        # Reshape Y - in the same way, except that the channel only has lev, not a window length
        def reshape_y(y):
            run, time, lev, Y, X = y.shape
            y_tmp = y.values.reshape(run * time, lev, Y, X)
            y_tmp = np.transpose(y_tmp, (0, 2, 3, 1)) # (population - Y - X - lev)
            return y_tmp.reshape(run * time, Y * X, lev)

        x_tensors_train.append(reshape_x(x_train_cut))
        x_tensors_test.append(reshape_x(x_test_cut))
        x_tensors_val.append(reshape_x(x_val_cut))

        for h in range(forecast_horizon):
            y_tensors_train[h].append(reshape_y(y_train_steps[h]))
            y_tensors_test[h].append(reshape_y(y_test_steps[h]))
            y_tensors_val[h].append(reshape_y(y_val_steps[h]))

        #storing metadata in lists for use in validation and testing of the results
        _, _, lev, Y, X = var_train.shape
        metadata_vars[var_name] = {
            'mean': var_mean.values.tolist(),
            'std': var_std.values.tolist()}

    #Create the actual tensors for Pytorch
    X_train = torch.tensor(np.concatenate(x_tensors_train, axis=-1), dtype=torch.float32)
    X_test = torch.tensor(np.concatenate(x_tensors_test, axis=-1), dtype=torch.float32)
    X_val = torch.tensor(np.concatenate(x_tensors_val, axis=-1), dtype=torch.float32)

    #The tensors for y are stacked according to the forecast horizon
    def stack_y(y_tensor_by_step):
        steps = []
        for h in range(forecast_horizon):
            step_h = np.concatenate(y_tensor_by_step[h], axis = -1)
            steps.append(step_h)
        return np.concatenate(steps, axis = -1)


    y_train = torch.tensor(stack_y(y_tensors_train), dtype=torch.float32)
    y_test  = torch.tensor(stack_y(y_tensors_test), dtype=torch.float32)
    y_val   = torch.tensor(stack_y(y_tensors_val), dtype=torch.float32)

    print(f'\nThe final shapes used for training:')
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
    print(f'X_val: {X_val.shape}, y_val: {y_val.shape}')

    n_vars_lev = len(variables) * lev #one variable yields to levels (lev 1 + 2), two variables yields four levels (var1: lev1 + lev2, var2: lev1+lev2) ie. four channels
    print(f'\nChannel in y: (n_vars_lev = {n_vars_lev}):')
    for h in range(forecast_horizon):
        print(f't+{h+1} : channels {h*n_vars_lev} - {(h+1)*n_vars_lev -1}')
    metadata = {
        'Y':             int(Y),
        'X':             int(X),
        'Lev':           int(lev),
        'window_length': int(window_length),
        'forecast_horizon' : int(forecast_horizon),
        'variables':     variables,
        'normalization': metadata_vars,
        'input_channels':  int(X_train.shape[-1]),
        'output_channels': int(y_train.shape[-1]),
        'n_vars_lev' : int(n_vars_lev),
    }

    return X_train, y_train, X_test, y_test, X_val, y_val, metadata


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def edges(Y, X):
    """
    Definition:
    This function creates the edge indexes to be used in the graph convolutional model
    Its an 8-neigbour edge index, where the nodes of the graph are connected to nodes on their imediate right, left, down and to their diagonals.
    Which nodes are connected to whom are all stored using the set() function to ensure there are no duplicates in the final output.

    Arguments: 
    arg[1] : Y - from the variable shape 
    arg[2] : X - from the variable shape

    Output:
    An edge index list decsribing which of the nodes in the graph that are connected. 
    """
    edge_set = set()

    for i in range(Y):
        for j in range(X):
            node = i * X + j

            #direct neighbours 
            if j < X - 1:
                edge_set.update([(node, node + 1), (node + 1, node)])
            if i < Y - 1:
                edge_set.update([(node, node + X), (node + X, node)])
            #diagonal 
            if i < Y - 1 and j < X - 1:
                edge_set.update([(node, node + X + 1), (node + X + 1, node)])
            if i < Y - 1 and j > 0:
                edge_set.update([(node, node + X - 1), (node + X - 1, node)])

    edge_index = torch.tensor(list(sorted(edge_set)), dtype=torch.long).t().contiguous()
    return edge_index

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def edges_with_weights(Y, X, sigma=1.0):
    """
    Definition:
    This function creates the edge indexes to be used in the graph convolutional model
    Its an 8-neigbour edge index, where the nodes of the graph are connected to nodes on their imediate right, left, down and to their diagonals.
    Which nodes are connected to whom are all stored using the set() function to ensure there are no duplicates in the final output.

    This updated function uses weights, where the weights are calculated using a Gaussian distance to calculate the weights between the nodes. 
    The weights for direct numbers is 1.0 unless changed, and weights for diagonal neighbours is calculated based on the distances. 

    Arguments: 
    arg[1] : Y - from the variable shape 
    arg[2] : X - from the variable shape
    arg[3] : Sigma (float) - sigma value for calculating the Gaussian weights. Default is 1.0 

    Output:
    An edge index list decsribing which of the nodes in the graph that are connected. 
    """
    edge_set = []
    weight_set = []

    for i in range(Y):
        for j in range(X):
            node = i * X + j

            neighbours = []
            if j < X - 1:
                neighbours.append((node + 1, 1.0)) #right
            if i < Y - 1:
                neighbours.append((node + X, 1.0)) #down
            if i < Y - 1 and j < X - 1:
                neighbours.append((node + X + 1, np.sqrt(2))) #diagonal right
            if i < Y - 1 and j > 0:
                neighbours.append((node + X - 1, np.sqrt(2))) #diagonal left 

            for neighbour, dist in neighbours:
                w = np.exp(-dist**2 / (2 * sigma**2)) #gaussian weight based on node distances - direct neighbours have weight = 1.0
                edge_set += [(node, neighbour), (neighbour, node)]
                weight_set += [w, w]

    seen, u_edges, u_weights = set(), [], []
    for e, w in zip(edge_set, weight_set):
        if e not in seen:
            seen.add(e)
            u_edges.append(e)
            u_weights.append(w)

    edge_index = torch.tensor(sorted(u_edges), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(
        [u_weights[u_edges.index(e)] for e in sorted(u_edges)],
        dtype=torch.float32
    )
    return edge_index, edge_weight


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def create_data(x_tensor, y_tensor, edge_index, edge_weight = None):
    """
    Definition:
    Creating the graph dataset. 
    The shape of the dataset should be (but are not limited to): (x, y, edge_index) - https://pytorch-geometric.readthedocs.io/en/1.4.2/modules/data.html 

    Arguments: 
    arg[1] : X_tensor - the tensor created and returned from the preprocessing
    arg[2] : Y_tensor - the tensor created and returned from the preprocessing
    arg[3] : Edge Index created by edges - 8 neighbourhood nodes in this case 
    arg[4] : Edge Weight (optional) - the weights for the edges

    Returns:
    The graph dataset to be used in training. 
    """
    dataset = []
    for i in range(x_tensor.shape[0]):
        graph = Data(x=x_tensor[i], y=y_tensor[i], edge_index=edge_index, edge_attr=edge_weight)
        dataset.append(graph)
    return dataset


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

class LaplacianLambdaMaxVal(BaseTransform):
    """ 
    Description: 
    Calulcates the largest eigenvalue of the graph Lalpacian, which is used for the Chebychev graph convolution model

    Arguments: 
    arg [1] : Basetransform (str) - Please choose the type of normalization. The default is symmetric normalization -> 'sym'. Other possibilities are:
    --> None : No normalization 
    For more information about these normalizations, please use the torch geometric website as linked below: (code inspired by the code in the link as well)
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/laplacian_lambda_max.html
    
    Returns:
    The data with the largest eigenvalue. This has been ran in a seperate testing notebook - largest eigenvalue is lambda = 1.526 for this dataset
    """
    def __init__(self, normalization='sym'):
        assert normalization in [None, 'sym'], 'Invalid normalization type'
        self.normalization = normalization

    def forward(self, data):
        from scipy.sparse.linalg import eigsh

        assert data.edge_index is not None
        num_nodes   = data.num_nodes
        edge_weight = data.edge_attr

        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(
            data.edge_index, edge_weight, self.normalization, num_nodes=num_nodes
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        data.lambda_max = lambda_max.real.item()
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(normalization={self.normalization})'


"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def dataloaders(train_graph, test_graph, val_graph, batch_size=32):
    """
    Definition:
    Creates the dataloaders for the training process.
    
    Arguments:
    arg[1] : Graph dataset used for training
    arg[2] : Graph dataset used for testing
    arg[3] : Graph dataset used for validation
    arg[4] : Batch size (int) - default 32

    outputs:
    The dataloaders
    """
    train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_graph,  batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_graph,   batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

