"""
plotting_scripts contains all plotting scripts for the inference results in the FYS5429 project. 

Figures:
[1] : val_loss - Training & validation loss per model 
[2] : rmse_per_channel - RMSE per forecast step and variable channel
[3] : forecast_panel - Forecasts for the models and the ground truth
[4] : scatter_residual - Scatter + residual plot
[5] : amplitude_collapse - standard deviation ratio across models and forecast length
[6] : kinetic_energy - KE-spektrum per model and channel
[7] : comp_cost - Memory and time diagnostics
[8] : dataviz  - distribution of data before/after spin-up period
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import cmocean
import json
import xarray as xr

#Please provide output directory file path and the filepath to the train file
output_directory  = 'filepath_save'   
data_path   = 'filepath_train.nc'  

# result_paths: please provide all file paths as a directory where the name given is used in the plots 
result_paths = {
    'FNO WeightedMSE' : 'filepath_to_fno_results',
    'Unet CombinedPhysics': 'filepath_to_unet_results',
    'WeightedChebGCN GradMSE': 'filepath_to_wchebgcn_results',
    'GCN GradMSE' : 'filepath_to_gcn_results',
}

# loss paths for the loss functions 
loss_paths= {
    'FNO WeightedMSE' : 'filepath_fno_results',
    'Unet CombinedPhysics' : 'filepath_unet_results',
    'WeightedChebGCN GradMSE': 'filepath_wchebgcn_results',
    'GCN GradMSE' : 'filepath_gcn_results',
}

# for the scatter plots 
scatter_model_path = 'filepath_results'
scatter_model_name = 'FNO WeightedMSE'

channel_names= ['ψ lev 1', 'ψ lev 2', 'q lev 1', 'q lev 2']
model_colors  = ['#7B3F9E', '#2E7D32', '#1565C0', '#9F3E3E']

# Computational cost — fill in numbers manually from training updates 
comp_cost = {
    'models' : ['FNO\n(Weighted MSE)', 'UNet\n(Physics Loss)',
                       'WeightedChebGCN\n(Grad MSE)', 'GCN\n(Grad MSE)'],
    'training_time' : [19.3, 16.0, 231, 132], # minutes
    'inference_time': [5.59, 5.1, 8.27, 4.33], # seconds
    'peak_memory' : [675.3, 585.5, 894.6, 791.2], # Megabytes
}

y_grid, x_grid = 64, 64
num_nodes = y_grid * x_grid
horizon   = 4
N_CH      = 4

os.makedirs(output_directory, exist_ok=True)


"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""


def _load(path):
    """Reshapes the predictions and targets to (N, horizon, nodes, channel)"""
    pred = np.load(os.path.join(path, 'predictions.npy'), allow_pickle=True)
    tgt  = np.load(os.path.join(path, 'targets.npy'),     allow_pickle=True)

    def _reshape(arr):
        if arr.ndim == 4 and arr.shape[1] == horizon and arr.shape[2] == num_nodes:
            return arr
        if arr.ndim == 4 and arr.shape[1] == num_nodes:
            return arr.transpose(0, 2, 1, 3)
        if arr.ndim == 3:
            N = arr.shape[0] // num_nodes
            return arr.reshape(N, num_nodes, horizon, N_CH).transpose(0, 2, 1, 3)
        if arr.ndim == 2:
            N = arr.shape[0] // num_nodes
            return arr.reshape(N, num_nodes, horizon, N_CH).transpose(0, 2, 1, 3)
        raise ValueError(f'Unknown shape: {arr.shape}')

    return _reshape(pred), _reshape(tgt)


"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""


def _load_loss(path):
    """loading loss_history.npy or history.csv (npy files are from GCNs and csv are from Unet and FNO)"""
    npy = os.path.join(path, 'loss_history.npy')
    csv = os.path.join(path, 'history.csv')
    if os.path.exists(npy):
        return np.load(npy, allow_pickle=True).item()
    if os.path.exists(csv):
        import pandas as pd
        df = pd.read_csv(csv)
        return {col: df[col].to_numpy(dtype=float)
                for col in df.columns if col != 'epoch'}
    raise FileNotFoundError(f'No loss function found in {path}')


"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def _save(fig, name, dpi=200):
    p = os.path.join(output_directory, name)
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    print(f'Saved: {p}')

"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_val_loss():
    model_names = list(loss_paths.keys())
    colors_map  = {n: model_colors[i % len(model_colors)]
                   for i, n in enumerate(model_names)}

    n = len(model_names)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()
    fig.suptitle('Training & Validation loss per epoch', fontsize=13)

    for i, (name, path) in enumerate(loss_paths.items()):
        ax = axes[i]
        color = colors_map[name]
        try:
            h = _load_loss(path)
            epochs = range(1, len(h['train_loss']) + 1)
            ax.plot(epochs, h['train_loss'], color=color, linewidth=1.8, label='Train')
            ax.plot(epochs, h['val_loss'],   color=color, linewidth=1.8,
                    linestyle='--', alpha=0.7, label='Val')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle='--')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, 'loss.png')
    return fig


"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_rmse_per_channel():
    steps = ['t+1', 't+2', 't+3', 't+4']
    fig, axes = plt.subplots(1, N_CH, figsize=(5 * N_CH, 4), sharey=False)
    fig.suptitle('RMSE per forecast step og channel', fontsize=13)

    for (name, path), color in zip(result_paths.items(),
                                   model_colors * 4):
        try:
            pred, tgt = _load(path)
            rmse = np.sqrt(np.mean((pred - tgt) ** 2, axis=(0, 2)))
            for c, ax in enumerate(axes):
                ax.plot(steps, rmse[:, c], marker='o', label=name,
                        color=model_colors[list(result_paths.keys()).index(name) % len(model_colors)],
                        linewidth=1.8)
        except Exception as e:
            print(f'Feil for {name}: {e}')

    for c, ax in enumerate(axes):
        ax.set_title(channel_names[c], fontsize=12)
        ax.set_xlabel('Forecast horizon' if c == N_CH - 1 else '')
        ax.set_ylabel('RMSE' if c == 0 else '')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = [Line2D([0], [0], color=model_colors[i], marker='o',
                      linewidth=1.8, label=n)
               for i, n in enumerate(result_paths.keys())]
    axes[0].legend(handles=handles, fontsize=8)

    plt.tight_layout()
    _save(fig, 'RMSEperforecastandchannel.png')
    return fig



"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_forecast_panel(sample=0, channel=0):
    ch_label = channel_names[channel]
    loaded = {n: _load(p) for n, p in result_paths.items()}
    model_names = list(result_paths.keys())

    n_steps = horizon
    n_rows  = 1 + len(model_names)

    cell_w, cell_h = 2.0, 1.8
    row_label_w = 2.2
    cbar_w = 0.18
    fw = row_label_w + cell_w * n_steps + cbar_w
    fh = cell_h * n_rows

    fig = plt.figure(figsize=(fw, fh))
    left   = row_label_w / fw
    right  = 1 - (cbar_w + 0.05) / fw
    gs = gridspec.GridSpec(n_rows, n_steps, left=left, right=right,
                           top=0.88, bottom=0.04, wspace=0.03, hspace=0.05)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_steps)]
                     for r in range(n_rows)])

    cax = fig.add_axes([right + 0.01, 0.12, 0.018, 0.72])

    first_tgt = loaded[model_names[0]][1]
    tgt_vals  = np.concatenate([first_tgt[sample, s, :, channel]
                                for s in range(horizon)])
    vmax = float(np.percentile(np.abs(tgt_vals), 98))
    imkw = dict(cmap=cmocean.cm.balance, origin='lower',
                vmin=-vmax, vmax=vmax, interpolation='nearest')

    def _label(ax, text):
        ax.annotate(text, xy=(-0.06, 0.5), xycoords='axes fraction',
                    xytext=(-4, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=12,
                    annotation_clip=False, linespacing=1.5)

    # Ground truth
    for col in range(n_steps):
        grid = first_tgt[sample, col, :, channel].reshape(y_grid, x_grid)
        axes[0, col].imshow(grid, **imkw)
        axes[0, col].set_title(f'$t+{col+1}$', fontsize=12, pad=3)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    _label(axes[0, 0], '$Y$')

    # Modeller
    mappable = None
    for ri, (name, (pred, _)) in enumerate(loaded.items(), start=1):
        for col in range(n_steps):
            grid = pred[sample, col, :, channel].reshape(y_grid, x_grid)
            mappable = axes[ri, col].imshow(grid, **imkw)
            axes[ri, col].set_xticks([]); axes[ri, col].set_yticks([])
        _label(axes[ri, 0], f'$\\hat{{Y}}$\n{name}')

    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(ch_label, fontsize=12)
    fig.suptitle(f'Predicted and ground truth sequence — {ch_label}',
                 fontsize=12, y=0.96)
    plt.tight_layout()
    _save(fig, f'newest_prediction_{ch_label.replace(" ", "_")}.png')
    return fig



"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_scatter_residual():
    from scipy.stats import pearsonr
    pred, tgt = _load(scatter_model_path)

    for step, step_label in [(0, 't+1'), (3, 't+4')]:
        p = pred[:, step, :, :].reshape(-1, N_CH)
        t = tgt[:,  step, :, :].reshape(-1, N_CH)
        rng = np.random.default_rng(42)
        idx = rng.choice(p.shape[0], size=min(5000, p.shape[0]), replace=False)

        fig, axes = plt.subplots(2, N_CH, figsize=(4.5 * N_CH, 8.5),
                                 gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
        fig.suptitle(f'{scatter_model_name}  ·  {step_label}', fontsize=13)

        for ch in range(N_CH):
            p_ch = p[idx, ch]; t_ch = t[idx, ch]; res = p_ch - t_ch

            # Scatter
            ax_sc = axes[0, ch]
            ax_sc.scatter(t_ch, p_ch, alpha=0.25, s=4, color='#7B3F9E', linewidths=0)
            lo, hi = min(t_ch.min(), p_ch.min()), max(t_ch.max(), p_ch.max())
            ax_sc.plot([lo, hi], [lo, hi], 'k-', linewidth=1.2, label='$y=x$')
            m, b = np.polyfit(t_ch, p_ch, 1)
            ax_sc.plot([lo, hi], [m*lo+b, m*hi+b], '--', color='#E53935',
                       linewidth=1.2, label=f'fit: {m:.2f}x + {b:.2e}')
            rmse_val = np.sqrt(np.mean(res**2))
            ax_sc.set_title(channel_names[ch], fontsize=12)
            ax_sc.set_xlabel('Target'); ax_sc.set_ylabel('Prediction')
            ax_sc.text(0.04, 0.93, f'RMSE={rmse_val:.4f}',
                       transform=ax_sc.transAxes, fontsize=12, va='top',
                       bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
            ax_sc.legend(fontsize=12); ax_sc.spines['top'].set_visible(False); ax_sc.spines['right'].set_visible(False)

            # Residual
            ax_res = axes[1, ch]
            ax_res.scatter(t_ch, res, alpha=0.25, s=4, color='#7B3F9E', linewidths=0)
            ax_res.axhline(0, color='k', linewidth=1.2, label='Zero error')
            ax_res.axhline(np.mean(res), color='#E53935', linewidth=1.2,
                           linestyle='--', label=f'Mean error: {np.mean(res):.2e}')
            ax_res.axhline(np.median(res), color='#FF8F00', linewidth=1.0,
                           linestyle=':', label=f'Median: {np.median(res):.2e}')
            ax_res.set_xlabel('Target'); ax_res.set_ylabel('Prediction − Target')
            ax_res.legend(fontsize=12); ax_res.spines['top'].set_visible(False); ax_res.spines['right'].set_visible(False)

        plt.tight_layout()
        fname = f'scatter_FNO_{step_label.replace("+", "")}.png'
        _save(fig, fname)
    return fig



"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_amplitude_collapse():
    steps = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(1, N_CH, figsize=(5 * N_CH, 4), sharey=True,
                             gridspec_kw={'wspace': 0.12})
    model_names = list(result_paths.keys())

    for mi, (name, path) in enumerate(result_paths.items()):
        color = model_colors[mi % len(model_colors)]
        pred, tgt = _load(path)
        for ch, ax in enumerate(axes):
            ratios = [np.std(pred[:, s, :, ch]) / max(np.std(tgt[:, s, :, ch]), 1e-12)
                      for s in range(horizon)]
            ax.plot(steps, ratios, marker='o', linewidth=2, color=color, label=name)

    for ch, ax in enumerate(axes):
        ax.axhline(1.0, color='black',   linewidth=1.4, linestyle='--', label='ideal (1.0)')
        ax.axhline(0.5, color='#E91E63', linewidth=1.0, linestyle=':',  label='collapse (0.5)')
        ax.set_title(f'Amplitude collapse — {channel_names[ch]}', fontsize=12)
        ax.set_xlabel('Forecast step')
        ax.set_xticks(steps)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    axes[0].set_ylabel(r'std ratio (pred / target)', fontsize=12)
    handles = ([Line2D([0],[0], color=model_colors[i], marker='o', linewidth=2, label=n)
                for i, n in enumerate(model_names)]
               + [Line2D([0],[0], color='black',   linewidth=1.4, linestyle='--', label='ideal (1.0)'),
                  Line2D([0],[0], color='#E91E63', linewidth=1.0, linestyle=':',  label='collapse (0.5)')])
    axes[0].legend(handles=handles, fontsize=12, loc='upper right')
    fig.suptitle('Amplitude collapse across forecast horizon', fontsize=13, y=1.02)

    plt.tight_layout()
    _save(fig, 'amplitudecollapse.png')
    return fig



"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_comp_cost():
    d = comp_cost
    x = np.arange(len(d['models']))
    w = 0.5
    colors = {'training': '#534AB7', 'inference': '#1D9E75', 'memory': '#D85A30'}

    fig = plt.figure(figsize=(12, 8))
    gs  = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    def _bar(ax, vals, color, ylabel):
        bars = ax.bar(x, vals, width=w, color=color, zorder=3, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels(d['models'], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ['top','right']: ax.spines[s].set_visible(False)
        ax.spines['left'].set_linewidth(0.5); ax.spines['bottom'].set_linewidth(0.5)
        offset = max(vals) * 0.02
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f'{val}', ha='center', va='bottom', fontsize=12, color='#444441')
        ax.set_ylim(0, max(vals) * 1.15)

    _bar(ax0, d['training_time'],  colors['training'],  'Training time [min]')
    _bar(ax1, d['inference_time'], colors['inference'], 'Inference time [sec]')
    _bar(ax2, d['peak_memory'],    colors['memory'],    'Peak memory [MB]')

    _save(fig, 'computational_cost.png')
    return fig


"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

def plot_dataviz(spinup=30):
    if not os.path.exists(data_path):
        print(f'data_path not found: {data_path} - data viz will not be plotted')
        return

    ds = xr.open_dataset(data_path, engine='netcdf4')
    variables = {'psi': 'Streamfunction Ψ', 'q': 'Potential Vorticity q'}
    nlev  = ds.sizes['lev']
    times = np.arange(ds.sizes['time'])
    colors_lev = ['tab:blue', 'tab:orange']

    fig, axes = plt.subplots(len(variables), 2, figsize=(22, 7 * len(variables)))
    if len(variables) == 1:
        axes = axes.reshape(1, 2)

    for row, (var, title) in enumerate(variables.items()):
        ax_h = axes[row, 0]; ax_e = axes[row, 1]
        for lev in range(nlev):
            color = colors_lev[lev % len(colors_lev)]
            layer = ds[var].isel(lev=lev)
            before = layer.values.ravel(); before = before[np.isfinite(before)]
            after  = layer.isel(time=slice(spinup, None)).values.ravel()
            after  = after[np.isfinite(after)]

            edges = np.linspace(min(before.min(), after.min()),
                                max(before.max(), after.max()), 201)
            ax_h.hist(before, bins=edges, density=True, histtype='step',
                      linewidth=2, color=color, label=f'Layer {lev+1} original')
            ax_h.hist(after,  bins=edges, density=True, histtype='step',
                      linewidth=2, linestyle='--', color=color,
                      label=f'Layer {lev+1} after spin-up')

            energy = (layer**2).sum(dim=('y','x'))
            mean   = energy.mean(dim='run').compute()
            ax_e.plot(times, mean, linewidth=2, color=color, label=f'Layer {lev+1}')

        ax_h.set_title(f'{title}\nDistribution before/after spin-up')
        ax_h.set_xlabel(var); ax_h.set_ylabel('Probability density')
        ax_h.grid(alpha=0.3); ax_h.legend()

        ax_e.axvline(spinup, color='red', linestyle='--', linewidth=2, label='Spin-up cutoff')
        ax_e.set_title(f'{title}\nEnergy evolution')
        ax_e.set_xlabel('Time index')
        ax_e.set_ylabel(f'Mean total squared {title}')
        ax_e.grid(alpha=0.3); ax_e.legend()

    fig.suptitle('Comparison of data before and after spin-up', y=1.02)
    plt.tight_layout()
    _save(fig, 'dataviz_spinup.png')
    return fig



"""
--------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    print('Storing all figures\n')
    plot_val_loss()
    plot_rmse_per_channel()
    plot_forecast_panel(sample=100, channel=0)
    plot_scatter_residual()
    plot_amplitude_collapse()
    plot_comp_cost()
    plot_dataviz()
    print('\n Figures are saved')
