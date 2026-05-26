import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from config import DATA_CFG

DATA_PATH   = DATA_CFG["data_path"]
WINDOW_SIZE = DATA_CFG["window_size"]
OUT_STEPS   = DATA_CFG["forecast_horizon"] 
INPUT_VARS  = DATA_CFG["input_vars"]
TARGET_VARS = DATA_CFG["target_vars"]
TRAIN_END   = DATA_CFG["train_end"]
VAL_END     = DATA_CFG["val_end"]


#extract and convert varibles names
def _parse_var_name(name):
    """Parse variable names from string into name and level. Split 'q_lev0' -> ('q', 0)
    
    Argument:
        name: "q_lev0", "q_lev1", "psi_lev0", "psi_lev1"

    Return:
        Two element tuple (variable name, and level index) 
    """
    parts = name.rsplit("_lev", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(
            f"Variable '{name}' must be changed to: 'q_lev0', 'q_lev1', 'psi_lev0', 'psi_lev1'."
        )
    return parts[0], int(parts[1])


def load_and_split(data_path, input_vars, target_vars):
    """
    Load the .nc file, build channels, split by run, and perform a z-score standardisation/normalisation

    Arguments:
        data_path: path to the .nc file
        input_var: ordered name list of input variables
        target_vars: ordered name list of target varaibles 
    
    Return:
    np.ndarray with normalised input and target array:
        x_train, x_val, x_test: (n_runs, time, n_input_vars,  H, W)
        y_train, y_val, y_test: (n_runs, time, n_target_vars, H, W)
    dict:
        norm_stats: x_mean, x_std, y_mean, y_std, input_vars, target_vars
    """
    ds = xr.open_dataset(data_path)

    def get_channel(name):
        """ 
        Extract variable level from .nc

        Argument:
        name: varaible name name_levelX formating 

        Return: 
        ndarray (n_runs, time, H, W) for given variable in argument
        """
        base, lev = _parse_var_name(name)
        if base not in ds.data_vars:
            raise ValueError(f"Variable not available: '{base}'. Use rather: {list(ds.data_vars)}")
        arr = ds[base].values  
        if lev >= arr.shape[2]:
            raise ValueError(f"Level {lev} not available for '{base}' (has {arr.shape[2]} levels).")
        return arr[:, :, lev, :, :] 

    #stack channels infot format -> (run, time, n_vars, H, W)
    inputs  = np.stack([get_channel(v) for v in input_vars],  axis=2).astype(np.float32)
    targets = np.stack([get_channel(v) for v in target_vars], axis=2).astype(np.float32)

    # Split by run (no temporal leakage, hence no shuffling)
    x_train, x_val, x_test = inputs[:TRAIN_END],  inputs[TRAIN_END:VAL_END],  inputs[VAL_END:]
    y_train, y_val, y_test = targets[:TRAIN_END], targets[TRAIN_END:VAL_END], targets[VAL_END:]

    # normalise using training statistics only, standarization by z-score
    def norm_stats_for(arr):
        """Compute z-score normalisation statstics for given array
        
        Argument:
            arr: ndarray with shape (n_runs, time, n_vars, H, W)
        
        Returns:
         ndarray: with shape (n_runs, time, n_vars, H, W) for the given variable

        """
        mean = arr.mean(axis=(0, 1, 3, 4), keepdims=True)
        std  = arr.std (axis=(0, 1, 3, 4), keepdims=True)
        std  = np.where(std == 0, 1.0, std)
        return mean, std

    x_mean, x_std = norm_stats_for(x_train)
    y_mean, y_std = norm_stats_for(y_train)
    
    
    x_train = (x_train - x_mean) / x_std
    x_val   = (x_val   - x_mean) / x_std
    x_test  = (x_test  - x_mean) / x_std
    
    
    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    norm_stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                      input_vars=list(input_vars), target_vars=list(target_vars))

    print(f"[dataset] inputs  shape : {x_train.shape}  vars={input_vars}")
    print(f"[dataset] targets shape : {y_train.shape}  vars={target_vars}")
    print(f"[dataset] window={WINDOW_SIZE}  out_steps={OUT_STEPS}")

    return x_train, x_val, x_test, y_train, y_val, y_test, norm_stats


class OceanDataset(Dataset):
    """
    Sliding-window samples:
    Iterate over all runs, extract input/target windows. In each sample, past frams from input pairs with future frames for target.
    The pair are reshaped into variable-major, where 
    Input layout for channels: channel= number of varibles * window_size + time_offset
    Target layout for channels: number of variables * out_step + time_offset

    Arguments:
    ndarray of normalized input/tragets:
        x_data: input array  shaped as: (n_runs, T, n_input_vars, H, W)
        y_data: output array shaped as: (n_runs, T, n_input_vars, H, W)
    integars:
        window: number of input time steps 
        out_steps:numbers of future time steps
    """

    def __init__(self, x_data, y_data, window, out_steps):
        n_runs, T, n_in,  H, W = x_data.shape
        _,      _, n_out, _, _ = y_data.shape

        xs, ys = [], []
        for r in range(n_runs):
            for t in range(T - window - out_steps + 1):
                
                # Input formating: (window, n_in,  H, W) -> (n_in, window, H, W) -> (n_in*window, H, W)
                x_win = x_data[r, t : t + window].transpose(1, 0, 2, 3)
                x_win = x_win.reshape(n_in * window, H, W)

                # Target formating : (out_steps,  n_out, H, W) -> (n_out, out_steps, H, W) -> (n_out*out_steps, H, W)
                y_win = y_data[r, t + window : t + window + out_steps].transpose(1, 0, 2, 3)
                y_win = y_win.reshape(n_out * out_steps, H, W)

                xs.append(x_win)
                ys.append(y_win)

        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.Y = torch.tensor(np.array(ys), dtype=torch.float32)
        print(f"[OceanDataset] X={self.X.shape}  Y={self.Y.shape}")



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_dataloaders(data_path=DATA_PATH, input_vars=INPUT_VARS, target_vars=TARGET_VARS,
                    window=WINDOW_SIZE, out_steps=OUT_STEPS,
                    batch_size=16, num_workers=0):
    """
    Build dataloader for train / val / test.

    Arguments:  
        data_path: path to .nc file
        input_vars: ordered name list string of input variable names 
        target_vars : ordered name list string of target variable names
        window : number of past timestep for each variable in input sample
        out_steps: number of future time steps for each variable in target
        batch_size : number of samples in a batch
        num_workers:  number of workers for data loading

    Return:
        train_loader: dataloader for train set
        val_loader: dataloafder for validation set
        test_loader: dataloader for test set
        in_channels : number of input_variables * window
        out_channels:number of input_variables * out_steps
        norm_stats: statistics for normalization

    """
    x_tr, x_va, x_te, y_tr, y_va, y_te, norm_stats = load_and_split(
        data_path, input_vars, target_vars
    )

    train_ds = OceanDataset(x_tr, y_tr, window, out_steps)
    val_ds   = OceanDataset(x_va, y_va, window, out_steps)
    test_ds  = OceanDataset(x_te, y_te, window, out_steps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    in_channels  = len(input_vars)  * window
    out_channels = len(target_vars) * out_steps

    return train_loader, val_loader, test_loader, in_channels, out_channels, norm_stats
