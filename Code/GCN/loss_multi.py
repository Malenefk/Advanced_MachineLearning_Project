import torch
import torch.nn.functional as F 

def mean_constraint(output, target, batch_index, num_channels):

    """
    Definition:
    Calculating a physics loss function, where a soft constraint is used to prevent the system from adding or removing energy from the system. 
    We want the model to learn that energy must either be generated or dissipated through energy transfer, not mechanically transferred into or out of a system. 
    The physics loss function is applied to a loss function calculating the loss per level

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - model truth
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : number of channels (int) - One variable has a channel 1 and 2. 
    
    Returns:
    The constrained physics loss (int) - (torch tensor)
    """

    unique_graphs = torch.unique(batch_index)
    constraint_losses = [[] for _ in range (num_channels)]
    for graph_id in unique_graphs:
        mask = batch_index == graph_id
        for c in range(num_channels):
            pred_mean = output[mask, c].mean()
            target_mean = target[mask, c].mean()
            constraint_losses[c].append((pred_mean - target_mean) **2)
    #Applied to each level for each variable:
    L_phys_per_channel = [torch.mean(torch.stack(constraint_losses[c])) for c in range(num_channels)]
    return torch.mean(torch.stack(L_phys_per_channel))

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def nodes_to_grid(x, batch_index, Y, X):
    
    """
    Definition:
    Creating a grid in terms of spatial coordinates (reversing the reshape function in preprocessing where we flattened the data)

    Arguments:
    arg[1] : x (torch tensor) - Node tensor with shape (N,C) -> C is the number of channels
    arg[2] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[3] : Y (int) - grid points in y-direction (64 in train.nc)
    arg[4] : X (int) - grid points in x-direction (64 in train.nc)

    Returns:
    A torch tensor with the shape: (Batch size, Channels, Y, X)
    """

    unique = torch.unique(batch_index)
    B = len(unique)
    C = x.shape[1]
    grids = []
    for graph_id in unique:
        mask = batch_index == graph_id
        grids.append(x[mask].reshape(Y,X,C).permute(2,0,1))
    return torch.stack(grids, dim = 0)

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def mse_baseline(output, target, batch_index, w = None, lambda_p = 0.01, **kwargs):
    
    """
    Definition:
    Calculating the Mean Squared Error loss from training data and target data. 

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - Punishment weight of the constrained physics loss   
    
    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """
    num_channels = output.shape[1]
    if w is None: 
        w = [1.0] * num_channels
    channel_losses = []
    for c in range(num_channels):
        channel_losses.append(torch.mean((output[:,c] - target[:,c]) **2)) 

    total_loss = sum(w[c] * channel_losses[c] for c in range(num_channels))
    L_phys = torch.tensor(0.0, device=output.device)
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# MSE with mean constraint 
def loss_function_with_constraint(output, target, batch_index, w=None, lambda_p=0.01, **kwargs):

    """
    Definition:
    Loss function including the physics informed constraint (lambda p)

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - Punishment weight of the constrained physics loss   

    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """

    num_channels = output.shape[1]
    if w is None:
        w = [1.0] * num_channels 
    channel_losses = []
    for c in range(num_channels):
        channel_losses.append(torch.mean((output[:,c] - target[:,c]) **2))

    L_phys = mean_constraint(output, target, batch_index, num_channels)
    weighted = sum(w[c] * channel_losses[c] for c in range(num_channels))
    total_loss = weighted + lambda_p * L_phys
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#weighted MSE
def weighted_mse(output, target, batch_index, w=None, lambda_p=0.01, **kwargs):

    """
    Definition:
    MSE loss function including the physics informed constraint (lambda p),
    and a weight for each channel calculated from the target variance. 
    W = 1 / sigma - high variance yields a lower weight value and low variance yields a higher weight value

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - Punishment weight of the constrained physics loss   

    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """
    
    num_channels = output.shape[1]

    channel_losses = []
    auto_w = []
    for c in range(num_channels):
        sigma2 = target[:,c].var().clamp(min=1e-6) #ensure min is large enough to avvoid division by zero
        loss_c = torch.mean((output[:,c] - target[:,c]) **2)
        channel_losses.append(loss_c)
        auto_w.append(1.0 / sigma2)

    L_phys = mean_constraint(output, target, batch_index, num_channels)
    weighted = sum(auto_w[c] * channel_losses[c] for c in range(num_channels))
    total_loss = weighted + lambda_p * L_phys
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#MSE with spatial gradient penalty
def mse_grad(output, target, batch_index, w=None, lambda_p=0.01, lambda_grad = 0.05, Y = 64, X = 64, **kwargs):
    
    """
    Definition:
    MSE loss function with a spatial gradient penalty applied - to punish the model for predicting to close to the mean value
    This is in order to prevent smoothing of the field and obey the model variance in the forecast field 

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - Punishment weight of the constrained physics loss
    arg[6] : lambda_grad (float) - punishment weight for prediciting close to the mean value
    arg[7] : Y (int) - grid points in y-direction (64 in train.nc)
    arg[8] : X (int) - grid points in x-direction (64 in train.nc)

    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """

    num_channels = output.shape[1]
    if w is None: 
        w = [1.0] * num_channels
    channel_losses = []
    for c in range(num_channels):
        channel_losses.append(torch.mean((output[:,c] - target[:,c]) **2))
    
    prediction_grid = nodes_to_grid(output, batch_index, Y, X)
    target_grid = nodes_to_grid(target, batch_index, Y, X)

    prediction_dx = prediction_grid[:,:,1:, :] - prediction_grid[:,:,:-1, :]
    prediction_dy = prediction_grid[:,:, :, 1:] - prediction_grid[:,:,:,:-1]
    target_dx = target_grid[:,:,1:, :] - target_grid[:,:,:-1, :]
    target_dy = target_grid[:,:, :, 1:] - target_grid[:,:,:,:-1]

    L_grad = F.mse_loss(prediction_dx, target_dx) + F.mse_loss(prediction_dy, target_dy)
    L_phys = mean_constraint(output, target, batch_index, num_channels)

    weighted = sum(w[c] * channel_losses[c] for c in range(num_channels))
    total_loss = weighted + lambda_p * L_phys + lambda_grad * L_grad
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#combined physics loss
def combined_physics(output, target, batch_index, w=None, lambda_p=0.01, lambda_grad = 0.05, lambda_std = 0.10, tail_weight = 0.5, Y = 64, X = 64, **kwargs):
    
    """
    Definition:
    MSE loss function with combined physics informed loss:
    - Mean constraint term preventing the model from adding or removing energy from the system
    - Spatial gradient penalty 
    - Variance penalty - Standard deviation penalty and tail weight - to retain variability and prevent smoothing of the forecast field 

    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - weight applied to the physics conservation constraint 
    arg[6] : lambda_grad (float) - weight applied to the spatial gradient penalty
    arg[7] : lambda_std (float) - weight applied to the variance penalty
    arg[8] : tail_weight (float) - weight which penalizes stronger anomalies more heavily, than the weaker anomalies
    arg[9] : Y (int) - grid points in y-direction (64 in train.nc)
    arg[10] : X (int) - grid points in x-direction (64 in train.nc)

    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """
        
    num_channels = output.shape[1]
    if w is None: 
        w = [1.0] * num_channels
    channel_losses = []
    for c in range(num_channels):
        anomaly_weight = 1.0 + tail_weight * (target[:,c].abs() > 1.0).float()
        loss_c = (anomaly_weight * (output[: , c] - target[:, c]) **2).mean()
        channel_losses.append(loss_c)

    prediction_grid = nodes_to_grid(output, batch_index, Y, X)
    target_grid = nodes_to_grid(target, batch_index, Y, X)

    prediction_dx = prediction_grid[:,:,1:, :] - prediction_grid[:,:,:-1, :]
    prediction_dy = prediction_grid[:,:, :, 1:] - prediction_grid[:,:,:,:-1]
    target_dx = target_grid[:,:,1:, :] - target_grid[:,:,:-1, :]
    target_dy = target_grid[:,:, :, 1:] - target_grid[:,:,:,:-1]

    L_grad = F.mse_loss(prediction_dx, target_dx) + F.mse_loss(prediction_dy, target_dy)

    L_phys = mean_constraint(output, target, batch_index, num_channels)
    prediction_std = output.std(dim=0)
    target_std = target.std(dim=0)
    Loss_std = F.mse_loss(prediction_std, target_std)

    weighted = sum(w[c] * channel_losses[c] for c in range(num_channels))
    total_loss = weighted + lambda_p * L_phys + lambda_grad * L_grad + lambda_std * Loss_std
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Graph laplacian
def graph_laplacian_loss(output, target, batch_index, w=None, lambda_p=0.01, lambda_lap = 0.05, edge_index = None, **kwargs):
    """
    Definition:
    Graph Laplacian loss function - to ensure the preservation of the graph structure in predictions. 
    Tested - but not used for final report. 
    For future work - could be interesting to explore further and combine with other physics informed losses + use on other GNNs
    
    Arguments:
    arg[1] : output (torch tensor) - model predictions 
    arg[2] : target (torch tensor) - target data
    arg[3] : Batch index (torch tensor) - vector that describes the relationship between nodes and graph
    arg[4] : w (int) - optional, a weight is added if we want to weigh the channel loss
    arg[5] : lambda_p (float) - Punishment weight of the constrained physics loss
    arg[6] : lambda_lap (float) - punishment weight for the graph Laplacian loss
    arg[7] : edge_index (optional) - edge index if we want to iterate through the graph edges

    Returns:
    [1] : Total loss
    [2] : Loss per channel 
    [3] : Physics informed loss
    """

    num_channels = output.shape[1]
    if w is None: 
        w = [1.0] * num_channels
    
    channel_losses = []
    for c in range(num_channels):
        channel_losses.append(torch.mean((output[:,c] - target[:,c]) **2))

    L_phys = mean_constraint(output, target, batch_index, num_channels)

    if edge_index is not None:
        src, dst = edge_index
        L_lap = torch.mean((output[src] - output[dst]) **2)
    else:
        L_lap = torch.tensor(0.0, device = output.device)

    weighted = sum(w[c] * channel_losses[c] for c in range(num_channels))
    total_loss = weighted + lambda_p * L_phys + lambda_lap * L_lap
    return total_loss, channel_losses, L_phys

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

Loss_register = {
    'mse_baseline' : mse_baseline,
    'mse_mean_constraint' : loss_function_with_constraint, 
    'weighted_mse' : weighted_mse,
    'mse_grad' : mse_grad,
    'combined_physics' : combined_physics,
    'graph_laplacian' : graph_laplacian_loss,
}

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Ensure that the user of the training chooses a valid loss function, options are listed in loss register above
def loss_multi(loss_name):
    if loss_name not in Loss_register:
        raise ValueError(f'Unkown Loss name: {loss_name}, please choose from the following loss functions: {Loss_register.keys()}')
    return Loss_register[loss_name]
    
    

