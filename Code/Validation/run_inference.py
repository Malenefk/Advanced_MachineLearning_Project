"""
run_inference.py

Runs  inference and saves the predictions.npy + targets.npy for all models.
The code works for all models: FNO/UNet and GCN-modeller. But because of different architecture during training, the model
must be specified for the code to run the right inference. 
The inference uses the same methods as found in the training loops :)

Inference results can be ran for all of the following model - as long as a model.pt path exists:
'FNO' — Fourier Neural Operator
'UNet' — UNet (CNN)
'GCN' — Simple GCN
'ChebGCN' — Chebyshev GCN
'WeightedChebGCN' — Weighted Chebyshev GCN
"""

import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

output_base = 'inference_results'

models_to_evaluate = [
    {
        'name'       : 'FNO WeightedMSE',
        'model_type' : 'FNO',
        'checkpoint' : 'filepath_fno/best_model.pt',
        'data_loader': 'torch',
        'extra'      : {},
    },
    {
        'name'       : 'Unet CombinedPhysics',
        'model_type' : 'UNet',
        'checkpoint' : 'filepath_unet/best_model.pt',
        'data_loader': 'torch',
        'extra'      : {},
    },
    {
        'name'       : 'WeightedChebGCN GradMSE',
        'model_type' : 'WeightedChebGCN',
        'checkpoint' : 'filepath_wchebgcn/best_model.pt',
        'data_loader': 'geo',
        'extra'      : {'K': 2, 'hidden_layers': 64},
    },
    {
        'name'       : 'GCN GradMSE',
        'model_type' : 'GCN',
        'checkpoint' : 'filepath_gcn/best_model.pt',
        'data_loader': 'geo',
        'extra'      : {'hidden_layers': 64},
    },]

test_dataset = 'path/to/test_dataset.pt'

# Hyperparameters 
input_channels  = 16   # window_length * n_vars_lev
output_channels = 4    # n_vars_lev
forecast_horizon = 4
window_length = 4
n_vars_lev = 4
batch_size = 32
use_delta = True 

# Lambda max is calculated already in the code folder for the GCNs
lambda_max = 1.526


def load_model(model_type, checkpoint_path, device, extra):
    """Importing the correct model architectures"""
    #####OBS - the model utilities file has to be downloaded in the same folder for this to work
    from models import ChebGCN, SimpleGCN, EdgeWeightChebGCN 

    k              = extra.get('K', 2)
    hidden         = extra.get('hidden_layers', 64)
    normalization  = extra.get('normalization', 'sym')

    if model_type == 'GCN':
        model = SimpleGCN(
            input_channel=input_channels,
            hidden_layers=hidden,
            output_channel=output_channels,
        )
    elif model_type == 'ChebGCN':
        model = ChebGCN(
            input_channel=input_channels,
            hidden_layers=hidden,
            output_channel=output_channels,
            K=k,
            normalization=normalization,
        )
    elif model_type == 'WeightedChebGCN':
        model = EdgeWeightChebGCN(
            input_channel=input_channels,
            hidden_layers=hidden,
            output_channel=output_channels,
            K=k,
            normalization=normalization,
        )
    elif model_type in ('FNO', 'UNet'):
        #OBS!! Once again - FNO and UNet models must be in the same folder for this to work :) 
        from fno_model import FNO2d
        from unet_model import UNet
        raise NotImplementedError(
            f' Please provide model type'
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f'  Downloaded {model_type} from {checkpoint_path}')
    return model


def slide_window(x_c, new_pred, n_vars_lev):
    return torch.cat([x_c[:, n_vars_lev:], new_pred], dim=1)


def get_pred_gcn(model_type, model, x_c, batch): 
    #forward pass
    if model_type == 'WeightedChebGCN':
        return model(x_c, batch.edge_index, lambda_max=lambda_max,
                     edge_weight=batch.edge_attr)
    elif model_type in ('ChebGCN',):
        return model(x_c, batch.edge_index, lambda_max=lambda_max)
    else:  # GCN
        return model(x_c, batch.edge_index)


def get_pred_cnn(model_type, model, x_c):
    #forward pass if model is UNet or FNO
    return model(x_c)


def run_inference_gcn(model, model_type, test_loader, device, out_dir):
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_c   = batch.x.clone()
            steps_pred, steps_tgt = [], []

            for step in range(forecast_horizon):
                pred_raw = get_pred_gcn(model_type, model, x_c, batch)
                gt_abs   = batch.y[:, step * n_vars_lev:(step + 1) * n_vars_lev]

                if use_delta:
                    x_current = x_c[:, -n_vars_lev:]
                    pred_abs  = x_current + pred_raw
                else:
                    pred_abs = pred_raw

                steps_pred.append(pred_abs.cpu())
                steps_tgt.append(gt_abs.cpu())
                x_c = slide_window(x_c, pred_abs, n_vars_lev)

            all_preds.append(torch.stack(steps_pred, dim=1))
            all_tgts.append(torch.stack(steps_tgt,  dim=1))

    predictions = torch.cat(all_preds, dim=0).numpy()
    targets     = torch.cat(all_tgts,  dim=0).numpy()
    _save_results(out_dir, predictions, targets)


def run_inference_cnn(model, model_type, test_loader, device, out_dir):
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # x_batch: (B, window*n_vars, H, W)
            # y_batch: (B, horizon*n_vars, H, W)
            x_c = x_batch.clone()
            steps_pred, steps_tgt = [], []

            for step in range(forecast_horizon):
                pred_raw = get_pred_cnn(model_type, model, x_c)
                gt_abs   = y_batch[:, step * n_vars_lev:(step + 1) * n_vars_lev]

                if use_delta:
                    x_current = x_c[:, -n_vars_lev:]
                    pred_abs  = x_current + pred_raw
                else:
                    pred_abs = pred_raw

                steps_pred.append(pred_abs.cpu())
                steps_tgt.append(gt_abs.cpu())

                x_c = torch.cat([x_c[:, n_vars_lev:], pred_abs.detach()], dim=1)

            all_preds.append(torch.stack(steps_pred, dim=1))
            all_tgts.append(torch.stack(steps_tgt,  dim=1))

    predictions = torch.cat(all_preds, dim=0).numpy()
    targets     = torch.cat(all_tgts,  dim=0).numpy()
    _save_results(out_dir, predictions, targets)


def _save_results(out_dir, predictions, targets):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(out_dir, 'targets.npy'),     targets)
    print(f'  Lagret predictions {predictions.shape} og targets {targets.shape} → {out_dir}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device found: {device}\n')

    test_dataset = torch.load(test_dataset)

    for cfg in models_to_evaluate:
        name       = cfg['name']
        model_type = cfg['model_type']
        ckpt       = cfg['checkpoint']
        loader_type= cfg['data_loader']
        extra      = cfg.get('extra', {})
        out_dir    = os.path.join(output_base, name.replace(' ', '_'))

        print(f'{name}')

        try:
            model = load_model(model_type, ckpt, device, extra)
        except NotImplementedError as e:
            print(f'{e}')
            continue

        if loader_type == 'geo':
            loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            run_inference_gcn(model, model_type, loader, device, out_dir)
        else:
            loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            run_inference_cnn(model, model_type, loader, device, out_dir)

        print()

    print('Inference is finished')
    print(f'Resulates are saved to: {output_base}/')