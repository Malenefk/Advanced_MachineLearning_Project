from loss_multi import loss_multi
import torch
from models import ChebGCN, SimpleGCN, EdgeWeightChebGCN, DeepChebGCN
import numpy as np
import os
import time 

#Moving one prediction ahead for each step in the forecast horizon - after each new prediction  
def slide_window_gcn(x_c, new_pred, window_length, lev):
    return torch.cat([x_c[:,lev:], new_pred], dim = 1)

#model input depending on the model name given in the run script 
def get_model_pred(model_name, model, x_c, batch):
    if model_name in ('ChebGCN', 'DeepChebGCN'):
        return model(x_c, batch.edge_index, lambda_max = 1.526)
    elif model_name == 'WeightedChebGCN':
        return model(x_c, batch.edge_index, lambda_max = 1.526, edge_weight = batch.edge_attr)
    else: #if model is GCN 
        return model(x_c, batch.edge_index)

def train(model_name, x_train_tensor, y_train_tensor, lambda_constraint, epoch_nr,
          train_loader, path, lr, val_loader, test_loader, K=2, normalization='sym',
          forecast_horizon = 4, ss_ratio = 0.4, ss_warmup_epochs = 10, use_var_penalty = False,
          lambda_var = 0.01, grad_clip = 1.0, use_scheduler = False, n_vars_lev=None, 
          loss_name = 'mse_mean_constraint', lambda_grad = 0.05, lambda_std = 0.10,
          lambda_lap = 0.05, use_edge_weights = False, use_delta = False):
    """
    Training loop for ChebGCN, DeepChebGCN, WeightedChebGCN and GCN.
    Compatible with any number of variables

    Arguments:
    arg[1] : model_name (str) - 'ChebGCN', 'GCN', 'DeepChebGCN' or 'WeightedChebGCN')
    arg[2] : x_train_tensor - Input data
    arg[3] : y_train_tensor - Target data 
    arg[4] : lambda_constraint (float) : Weight for physical constraint loss
    arg[5] : epoch_nr (int) - Number of epochs
    arg[6] : train_loader - DataLoader for training set
    arg[7] : val_loader - DataLoader for validation set
    arg[8] : test_loader - DataLoader for test set
    arg[9] : path (str)  - Directory to save model and outputs
    arg[10] : lr (float) - Learning rate
    arg[11] : K (int) - Chebyshev polynomial degree, K = 2 for the GCN model 
    arg[12] : normalization (str) - Graph Laplacian normalization for ChebGCN, standard is 'sym'
    arg[13] : forecast_horizon (int) - Number of steps to predict ahead
    arg[14] : ss_ratio (float) - Scheduled sampling ratio [0,1]  
    arg[15] : ss_warmup_epochs (int) - Epochs to warm up scheduled sampling
    arg[16] : use_var_penalty (Boolean) - Whether or not to include variance penalty in the loss function
    arg[17] : lambda_var (float) - Weight for variance penalty in the loss
    arg[18] : grad_clip (float) - Maximum value for gradient clipping [0,1]
    arg[19] : use_scheduler (Boolean) - Whether to use learning rate scheduler based on validation loss
    arg[20] : n_vars_lev (int or None) - number of variables and levels (if None -> number of levels is extracted from number of variables)
    arg[21] : loss_name (str) - 'mse_baseline', 'mse_mean_constraint', 'weighted_mse', 'mse_grad' and 'combined_physics' 
    arg[22] : lambda_grad (float) - Spatial gradient penalty 
    arg[23] : lambda_std (float) - Standard deviation penalty 
    arg[24] : lambda_lap (float) - Weight for Laplacian penalty
    arg[25] : use_edge_weights (Boolean) - Whether to use edge weights in the model (only applicable for WeightedChebGCN)
    arg[26] : use_delta (Boolean) - Whether to use delta forecasting (predicting changes instead of absolute values)
    """
    os.makedirs(path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Tracking training time and memory usage
    training_start = time.time()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'The GPU used for training is: {gpu_name}, and the memory used is: {gpu_mem} ')
    else:
        print(f'Running on CPU')

    
    loss_fn = loss_multi(loss_name)
    print(f'Loss function : {loss_name}, Delta Forecasting {use_delta}')
    #Automatic calculation of n_vars_lev if not given as an input
    if n_vars_lev is None:
        n_vars_lev = y_train_tensor.shape[2] // forecast_horizon
    lev = n_vars_lev
    window_length = x_train_tensor.shape[2] // n_vars_lev

    if model_name == 'ChebGCN':
        model = ChebGCN(
            input_channel=x_train_tensor.shape[2],
            hidden_layers=64,
            output_channel=n_vars_lev,
            K=int(K),
            normalization=normalization)
    elif model_name == 'GCN':
        model = SimpleGCN(
            input_channel=x_train_tensor.shape[2],
            hidden_layers=64,
            output_channel=n_vars_lev)
    elif model_name == 'DeepChebGCN':
        model = DeepChebGCN(
            input_channel= x_train_tensor.shape[2],
            hidden_layers=64,
            output_channel=n_vars_lev,
            K = int(K), 
            normalization= normalization,
            block_numbers= 4)
    elif model_name == 'WeightedChebGCN':
        model = EdgeWeightChebGCN(
            input_channel= x_train_tensor.shape[2],
            hidden_layers=64,
            output_channel=n_vars_lev,
            K = int(K), 
            normalization= normalization)
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'. Please choose betweeen: 'ChebGCN' , 'GCN' , 'DeepChebGCN', 'WeightedChebGCN'.")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Learning rate scheduler with a patience of 5 epochs - ie. 5 epochs with no improvement of the validation loss iniaties the learning rate scheduler
    #https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= 5, factor = 0.5, verbose = True)
        
    #SS ratio - the model starts with a higher value of SS_ratio and takes a number of ss_warmup_epochs to reach the full ss_ratio for autoregressive forecasting 
    def get_ss_ratio(epoch):
        ss_warmup = int(ss_warmup_epochs)
        fh = int(forecast_horizon)
        if ss_warmup <= 0 or fh == 1:
            return ss_ratio 
        return min(ss_ratio, ss_ratio * epoch / ss_warmup) 

    history = {
        'train_loss': [], 'train_channel_losses': [], 'train_constraint_loss': [],
        'val_loss':   [], 'val_channel_losses':   [], 'val_constraint_loss':   []}
    
    best_val_loss = float('inf')
    patience = 10
    epochs_with_no_improvement = 0
    min_delta = 1e-3 

    for epoch in range(epoch_nr):
        current_ss = get_ss_ratio(epoch + 1)
        #training starts:
        model.train()
        run_loss = 0.0
        run_phys = 0.0
        run_channel = [0.0] * n_vars_lev

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            x_c = batch.x.clone()
            total_step_loss = torch.tensor(0.0, device = device)

            for step in range(forecast_horizon):
                pred_orig = get_model_pred(model_name, model, x_c, batch)
                t_abs = batch.y[:,step * n_vars_lev : (step +1) * n_vars_lev] #ensures the right target y
                
                if use_delta:  
                    x_current = x_c[:, -n_vars_lev:]
                    gt = t_abs - x_current #calculates the change from t --> t+1
                    pred_abs = x_current + pred_orig #predicts the change for the next time step   
                else:
                    gt = t_abs 
                    pred_abs = pred_orig

                var_loss = torch.tensor(0.0, device = device)
                if use_var_penalty:
                    var_loss = lambda_var * torch.mean((pred_orig.var(dim=0) - gt.var(dim=0)) **2)

                step_loss, channel_losses, phys = loss_fn(pred_orig, gt, batch.batch, lambda_p=lambda_constraint, lambda_grad = lambda_grad, lambda_std = lambda_std, lambda_lap = lambda_lap, edge_index = batch.edge_index,)
                
                step_loss = step_loss + var_loss
                step_weight = 1.0 + step / max(1, forecast_horizon-1)
                total_step_loss = total_step_loss + step_weight * step_loss

                run_phys += phys.item()
                for c, cl in enumerate(channel_losses):
                    run_channel[c] += cl.item()

                use_pred = torch.rand(1).item() < current_ss
                next_input = pred_abs.detach() if use_pred else t_abs
                x_c = slide_window_gcn(x_c, next_input, window_length, lev)

            weight_sum = sum(1.0 + s / max(1, forecast_horizon - 1) for s in range(forecast_horizon))
            loss = total_step_loss / weight_sum
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip)
            optimizer.step()

            run_loss += loss.item()

        n = len(train_loader)
        history['train_loss'].append(run_loss / n)
        history['train_constraint_loss'].append(run_phys / n * forecast_horizon)
        history['train_channel_losses'].append([v / n * forecast_horizon for v in run_channel])

        #validation
        model.eval()
        val_loss = 0.0
        val_phys = 0.0
        val_channel = [0.0] * n_vars_lev

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_c = batch.x.clone()
                total_step_loss = torch.tensor(0.0, device = device)

                for step in range(forecast_horizon):
                    pred_orig = get_model_pred(model_name, model, x_c, batch)
                    t_abs = batch.y[:,step * n_vars_lev : (step +1) * n_vars_lev]
                    if use_delta: 
                        x_current = x_c[:, -n_vars_lev:]
                        gt = t_abs - x_current 
                        pred_abs = x_current + pred_orig 
                    else:
                        gt = t_abs 
                        pred_abs = pred_orig                    
                    v_loss, v_ch_losses, v_phys = loss_fn(pred_orig, gt, batch.batch, lambda_p=lambda_constraint, lambda_grad = lambda_grad, lambda_std = lambda_std, lambda_lap = lambda_lap, edge_index = batch.edge_index,)
                    step_weight = 1.0 + step / max(1, forecast_horizon - 1)
                    total_step_loss = total_step_loss  + step_weight * v_loss 

                    val_phys += v_phys.item()
                    for c, cl in enumerate(v_ch_losses):
                        val_channel[c] += cl.item()
                
                    x_c = slide_window_gcn(x_c, pred_abs, window_length, lev)
                        
                weight_sum = sum(1.0 + s  / max(1, forecast_horizon - 1) for s in range(forecast_horizon))
                val_loss += (total_step_loss / weight_sum).item()
                

        nv = len(val_loader)
        epoch_val_loss = val_loss / nv
        history['val_loss'].append(epoch_val_loss)
        history['val_constraint_loss'].append(val_phys / (nv * forecast_horizon))
        history['val_channel_losses'].append([v / (nv * forecast_horizon) for v in val_channel])

        ss_str = f' ss={current_ss:.3f}' if forecast_horizon > 1 else ''
        train_ch_str = '  '.join(
            [f'ch{c}={history["train_channel_losses"][-1][c]:.4f}' for c in range(n_vars_lev)]
        )
        print(
            f"Epoch {epoch+1:>3}/{epoch_nr} | "
            f"Train {history['train_loss'][-1]:.4f} ({train_ch_str}) | "
            f"Val {epoch_val_loss:.4f}{ss_str}")
        
        if use_scheduler:
            scheduler.step(epoch_val_loss)
        
        #saves only if validation loss reaches a new best value
        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            epochs_with_no_improvement = 0 
            torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))
            print(f"  ✓ New best val loss: {best_val_loss:.4f} — model saved")
        else:
            #Counts number of epochs with no validation loss improvement - after a patience of 10 the traning is stopped early on
            epochs_with_no_improvement += 1
            if epochs_with_no_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    #test
    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt'), map_location=device))
    model.eval()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_c = batch.x.clone()
            steps_pred = []
            steps_t = []
            for step in range(forecast_horizon):
                pred_orig = get_model_pred(model_name, model, x_c, batch)
                t_abs = batch.y[:,step * n_vars_lev : (step +1) * n_vars_lev]
                if use_delta: 
                    x_current = x_c[:, -n_vars_lev:]
                    pred_abs = x_current + pred_orig 
                else:
                    pred_abs = pred_orig
                steps_pred.append(pred_abs.cpu())
                steps_t.append(t_abs.cpu())

                x_c = slide_window_gcn(x_c, pred_abs, window_length, lev)
        
            all_predictions.append(torch.stack(steps_pred, dim = 1))
            all_targets.append(torch.stack(steps_t, dim = 1))

    final_predictions = torch.cat(all_predictions, dim=0).numpy()
    final_targets    = torch.cat(all_targets, dim=0).numpy()

    #saving all final predictions, targets and loss history as numpy arrays for validation plotting
    np.save(os.path.join(path, 'predictions.npy'), final_predictions)
    np.save(os.path.join(path, 'targets.npy'), final_targets) 
    np.save(os.path.join(path, 'loss_history.npy'), history)

    training_end = time.time() - training_start
    if torch.cuda.is_available():
        print(f'Training Time: {training_end}s on {gpu_name} ({gpu_mem:.3f} GB)')


    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {path}")
    print(f"Shape of predictions: {final_predictions.shape}")
    print(f"Shape of targets: {final_targets.shape}")
    return history, best_val_loss
