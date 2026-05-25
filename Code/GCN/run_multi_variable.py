#Import necessary packages
import os
import json
import itertools

from preprocess_multi import preprocess_variable, edges, create_data, dataloaders
from train_multi import train

"""
Submit script for training.
All variables and function parameters are chosen here - such that all other scripts dont need to be changed when finished. 
"""

#Saving the results and datapath to training dataset 
data_path    = '/itf-fi-ml/home/malenefk/train.nc'
result_path = '/itf-fi-ml/home/malenefk/Advanced_ML/Mallis/results/multi_variable_TPE/gcn'

#Selecting variables, window length, whether to use spin-up or not and the batch size 
variables     = ['psi', 'q']
window_length = 4
batch_size    = 32
spin_up       = 30

#Sweep through all hyperparameters parameters for easy tuning and testing of the models 
SWEEP = {
    'model_name':    ['WeightedChebGCN'],
    'epochs':        [100],
    'lr':            [1e-3],
    'lambda_p':      [0.05],
    'K':             [3],
    'normalization': ['sym'],

    #to initziate autoregressive rollout with ss ratio
    'forecast_horizon': [4], 
    'ss_ratio' : [0.5],
    'ss_warmup_epochs' : [10], 

    #Loss functions and penalty
    'lambda_var': [0.01],
    'grad_clip' : [1.0],
    'use_scheduler' : [True],
    'loss_name' : ['mse_mean_constraint', 'weighted_mse'],
    'lambda_grad' : [0.075],
    'lambda_std' : [0.1],
    'lambda_lap' : [0.1], 
    'use_edge_weights' : [True],
    'use_delta' : [True],}

print('=' * 60)
print(f'Preprocessing variables : {variables}')
print('=' * 60)

X_train, y_train, X_test, y_test, X_val, y_val, metadata = preprocess_variable(
    data_path=data_path,
    variable=variables,
    window_length=window_length,
    spin_up=spin_up,
    forecast_horizon=4,)

print(f'\nInput channels  (X): {X_train.shape[-1]}')
print(f'Output channels (y): {y_train.shape[-1]}')

from preprocess_multi import edges_with_weights 
edge_index, edge_weight = edges_with_weights(metadata['Y'], metadata['X'])
train_graph = create_data(X_train, y_train, edge_index, edge_weight = edge_weight)
test_graph = create_data(X_test, y_test, edge_index , edge_weight = edge_weight)
val_graph = create_data(X_val, y_val, edge_index, edge_weight = edge_weight)

train_loader, test_loader, val_loader = dataloaders(
    train_graph, test_graph, val_graph, batch_size=batch_size)

os.makedirs(result_path, exist_ok=True)
with open(os.path.join(result_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=4) #creates a file containing all metadata

#combine all hyperparameters from SWEEP
keys   = list(SWEEP.keys())
combos = [dict(zip(keys, v)) for v in itertools.product(*SWEEP.values())]

#If K != 2 in SWEEP, this iteration sets K = 2 if model is given as 'GCN'
unique_combos, seen = [], set()
for cfg in combos:
    if cfg['model_name'] == 'GCN':
        cfg['K'] = 2
    key = json.dumps(cfg, sort_keys=True)
    if key not in seen:
        seen.add(key)
        unique_combos.append(cfg)

print(f'\n{len(unique_combos)} experiments to run\n')


#Sum up hyperparameters used
summary = []

for i, cfg in enumerate(unique_combos):
    mode_str = f"ar{cfg['forecast_horizon']}" if cfg['forecast_horizon'] > 1 else 'onestep'
    exp_name = (
        f"{cfg['model_name']}"
        f"_lr{cfg['lr']}"
        f"_ep{cfg['epochs']}"
        f"_lp{cfg['lambda_p']}"
        f"_K{cfg['K']}"
        f"_{cfg['loss_name']}"
        f"_lgrad{cfg['lambda_grad']}"
        f"_lstd{cfg['lambda_std']}"
        f"_llap{cfg['lambda_lap']}"
        f"_edge_weight{cfg['use_edge_weights']}"
        f"_use_delta{cfg['use_delta']}"
        f"_{mode_str}"
    )
    exp_path = os.path.join(result_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    print("=" * 60)
    print(f"[{i+1}/{len(unique_combos)}] {exp_name}")
    print("=" * 60)

    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump({**cfg, 'variables': variables}, f, indent=4)

    try:
        history, best_val = train(
            model_name=cfg['model_name'],
            x_train_tensor=X_train,
            y_train_tensor=y_train,
            lambda_constraint=cfg['lambda_p'],
            epoch_nr=cfg['epochs'],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            path=exp_path,
            lr=cfg['lr'],
            K=cfg['K'],
            normalization=cfg['normalization'],
            forecast_horizon = cfg['forecast_horizon'],
            ss_ratio = cfg['ss_ratio'],
            ss_warmup_epochs = cfg['ss_warmup_epochs'],
            lambda_var = cfg['lambda_var'],
            grad_clip = cfg['grad_clip'],
            use_scheduler = cfg['use_scheduler'],
            loss_name = cfg['loss_name'],
            lambda_grad = cfg['lambda_grad'],
            lambda_std = cfg['lambda_std'],
            lambda_lap = cfg['lambda_lap'],
            use_edge_weights = cfg['use_edge_weights'],
            use_delta = cfg['use_delta'],
        )
        summary.append({'experiment': exp_name, 'config': cfg, 'best_val_loss': best_val, 'status': 'ok'})

    except Exception as e:
        import traceback
        traceback.print_exc() #returns the traceback of the error so its easier to debug 
        print(f"  ✗ FEIL: {e}")
        summary.append({'experiment': exp_name, 'config': cfg, 'best_val_loss': None, 'status': f'error: {e}'})

#If the training is successful, the summary is sorted based on the validation loss and contains a summary of all experiments for easy overview and comparison as a json file
successful = sorted([s for s in summary if s['status'] == 'ok'], key=lambda s: s['best_val_loss'])

print('\n' + '=' * 60)
print('SWEEP is finished')
print('=' * 60)
for rank, s in enumerate(successful, 1):
    print(f"  #{rank:>2}  val={s['best_val_loss']:.6f}  {s['experiment']}")

with open(os.path.join(result_path, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)

print(f'\nSummary is saved to: {result_path}/summary.json')
