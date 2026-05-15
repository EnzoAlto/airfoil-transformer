import h5py
import zipfile
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import math
import rounders
import tensorflow as tf
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.ndimage import rotate, distance_transform_edt



# ── Data Loading ────────────────────────────────────────────────────────────────

def unzip_airfoil_data(
    zip_path='data/airfoil_1k_data_rng.zip',
    extract_to='data/airfoil_1k_data_rng',
):
    """Extract the airfoil dataset zip file and return the output directory."""
    zip_path = Path(zip_path).expanduser()
    extract_to = Path(extract_to).expanduser()
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    print(f'Extracted {zip_path} to {extract_to}')
    return extract_to


def load_shape(data_path, key, use_dask=True, chunks=500):
    """Load a shape dataset ('landmarks', 'grassmann', 'cst', 'bezier')."""
    with h5py.File(data_path, 'r') as hf:
        ds = hf['shape'][key]
        return da.from_array(ds, chunks=chunks).compute() if use_dask else ds[()]


def load_aero_coeffs(data_path, alpha='alpha+04', use_dask=True, chunks=500):
    """Return (C_l, C_d, C_m) arrays for the given angle-of-attack group."""
    with h5py.File(data_path, 'r') as hf:
        grp = hf[alpha]
        if use_dask:
            cl = da.from_array(grp['C_l'], chunks=chunks).compute()
            cd = da.from_array(grp['C_d'], chunks=chunks).compute()
            cm = da.from_array(grp['C_m'], chunks=chunks).compute()
        else:
            cl, cd, cm = grp['C_l'][()], grp['C_d'][()], grp['C_m'][()]
    return cl, cd, cm


def load_flow_field(data_path, idx, alpha='alpha+04'):
    """Return dict of flow-field arrays (x, y, rho, rho_u, rho_v, e, omega) for one airfoil."""
    with h5py.File(data_path, 'r') as hf:
        ff = hf[alpha]['flow_field']['{:04d}'.format(idx)]
        return {k: ff[k][()] for k in ff.keys()}


# ── Geometry Calculations ────────────────────────────────────────────────────────

def compute_thickness_camber(landmarks):
    """
    Given landmarks array (N, 1001, 2), return (thickness_2d, camber_2d, x_c).

    thickness_2d: (N, 500) absolute thickness at each x/c station
    camber_2d:    (N, 500) camber (midpoint) at each x/c station
    x_c:          (500,)   chord-normalised x stations from 0.002 to 1.0
    """
    thickness_2d = abs(landmarks[:, 501:, 1] - np.flip(landmarks[:, :500, 1], axis=1))
    camber_2d = (landmarks[:, 501:, 1] + np.flip(landmarks[:, :500, 1], axis=1)) / 2
    x_c = np.arange(0.002, 1.002, 0.002)
    return thickness_2d, camber_2d, x_c


def thickness_camber_dataframes(thickness_2d, camber_2d, x_c):
    """Return (thickness_df, camber_df) long-form DataFrames for seaborn lineplot."""
    N = thickness_2d.shape[0]
    x_rep = np.tile(x_c, N)
    thickness_df = pd.DataFrame({'x': x_rep, 'thickness': thickness_2d.ravel()})
    camber_df = pd.DataFrame({'x': x_rep, 'camber': camber_2d.ravel()})
    return thickness_df, camber_df


# ── PCA Helpers ──────────────────────────────────────────────────────────────────

def pca_reduce(data, n_components, return_scaler=False):
    """Standardise data then apply PCA. Returns transformed array (and optionally scaler)."""
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data_std)
    return (reduced, scaler) if return_scaler else reduced


def pca_variance_curve(data, max_components=None):
    """
    Plot cumulative explained-variance vs n_components and return the ratios list.
    Prints explained variance for n_components < 6.
    """
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    max_components = max_components or data.shape[1]
    n_comps = np.arange(max_components) + 1
    var_ratio = []
    for i in n_comps:
        pca = PCA(n_components=i)
        pca.fit(data_std)
        var_ratio.append(np.sum(pca.explained_variance_ratio_))
        if i < 6:
            print(f'n_components={i}  explained_variance={pca.explained_variance_}')

    plt.figure(figsize=(4, 2))
    plt.grid()
    plt.plot(n_comps, var_ratio, marker='o')
    plt.xlabel('n_components')
    plt.ylabel('Explained variance ratio')
    plt.xticks([int(i) for i in n_comps])
    plt.tight_layout()
    plt.show()
    return var_ratio


# ── Rasterisation ───────────────────────────────────────────────────────────────

def _rotate_landmarks(landmarks, angle_deg, pivot=(0.25, 0.0)):
    """Clockwise rotation around the quarter-chord (centre of pressure) by angle_deg."""
    a = np.radians(-angle_deg)  # negative = clockwise
    cos_a, sin_a = np.cos(a), np.sin(a)
    x = landmarks[:, :, 0] - pivot[0]
    y = landmarks[:, :, 1] - pivot[1]
    x_rot = x * cos_a - y * sin_a + pivot[0]
    y_rot = x * sin_a + y * cos_a + pivot[1]
    return np.stack([x_rot, y_rot], axis=-1)


def _rasterise(landmarks, resolution, xs, ys):
    from matplotlib.path import Path
    grid_x, grid_y = np.meshgrid(xs, ys)
    query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grids = np.zeros((len(landmarks), resolution, resolution), dtype=np.uint8)
    for i, lm in enumerate(landmarks):
        grids[i] = Path(lm).contains_points(query_pts).reshape(resolution, resolution)
    return grids


def _airfoil_interior_mask(landmark, grid_x, grid_y):
    """Return True for grid cells inside the airfoil landmark polygon."""
    from matplotlib.path import Path
    query_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return Path(landmark).contains_points(query_pts).reshape(grid_x.shape)


def landmarks_to_grid(landmarks, resolution=128, x_range=(0.0, 1.0), y_range=(-0.5, 0.5), padding=0.0, save_path=None):
    """
    Rasterise airfoil landmark outlines into square binary images at 0, 4, and 12 deg AOA.

    Parameters
    ----------
    landmarks  : (N, 1001, 2) array of airfoil outlines
    resolution : int — side length of the square grid in pixels (default 128)
    x_range    : (xmin, xmax) chord extent of the grid
    y_range    : (ymin, ymax) normal extent — keep square: y span == x span
    padding    : float — extra space added to all four sides in data units (default 0.0)
    save_path  : optional str path to save as .h5 (e.g. 'data/grids_128.h5')
                 saves datasets: 'alpha0', 'alpha4', 'alpha12'

    Returns
    -------
    dict with keys 'alpha0', 'alpha4', 'alpha12', each (N, resolution, resolution) uint8
    """
    xs = np.linspace(x_range[0] - padding, x_range[1] + padding, resolution)
    ys = np.linspace(y_range[0] - padding, y_range[1] + padding, resolution)

    result = {
        'alpha0':  _rasterise(landmarks, resolution, xs, ys),
        'alpha4':  _rasterise(_rotate_landmarks(landmarks,  4), resolution, xs, ys),
        'alpha12': _rasterise(_rotate_landmarks(landmarks, 12), resolution, xs, ys),
    }

    if save_path is not None:
        with h5py.File(save_path, 'w') as hf:
            for key, grids in result.items():
                hf.create_dataset(key, data=grids, compression='gzip')
        print(f'Saved {len(landmarks)} airfoils × 3 AOAs to {save_path}')

    return result


def flow_field_to_grid(flow_dict, resolution=128, xlim=(-0.25, 1.25), ylim=(-0.5, 0.5),
                       metrics=('rho', 'rho_u', 'rho_v', 'e', 'omega'),
                       method='linear', fill_nearest=True, airfoil_landmark=None,
                       airfoil_fill_value=np.nan, dtype=np.float32):
    """
    Interpolate one scattered flow-field dict onto fixed 2D raster grids.

    Parameters
    ----------
    flow_dict    : dict returned by load_flow_field
    resolution   : int or (height, width) output grid size
    xlim, ylim   : plotting/interpolation bounds in flow-field coordinates
    metrics      : flow variables to rasterise
    method       : scipy griddata method: 'linear', 'nearest', or 'cubic'
    fill_nearest : fill NaNs from linear/cubic interpolation with nearest values
    airfoil_landmark : optional (N, 2) landmark polygon for blanking the solid body
    airfoil_fill_value : value assigned inside the airfoil, default NaN
    dtype        : output array dtype

    Returns
    -------
    dict with keys for each metric, each array shape (height, width)
    """
    if isinstance(resolution, int):
        height = width = resolution
    else:
        height, width = resolution

    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    airfoil_mask = None
    if airfoil_landmark is not None:
        airfoil_mask = _airfoil_interior_mask(airfoil_landmark, grid_x, grid_y)

    x = flow_dict['x']
    y = flow_dict['y']
    in_bounds = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    points = np.column_stack([x[in_bounds], y[in_bounds]])

    if len(points) == 0:
        raise ValueError('No flow-field points found inside xlim/ylim.')

    grids = {}
    for metric in metrics:
        values = flow_dict[metric][in_bounds]
        grid = griddata(points, values, (grid_x, grid_y), method=method)

        if fill_nearest and np.isnan(grid).any() and method != 'nearest':
            nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid = np.where(np.isnan(grid), nearest, grid)

        if airfoil_mask is not None:
            grid[airfoil_mask] = airfoil_fill_value

        grids[metric] = grid.astype(dtype)

    return grids


def _flow_field_grid_worker(args):
    """Rasterise all requested alphas/metrics for one airfoil index."""
    (
        data_path, out_i, src_idx, alphas, resolution, xlim, ylim, metrics,
        method, fill_nearest, mask_airfoil, landmark_key, airfoil_fill_value, dtype
    ) = args

    key = f'{src_idx:04d}'
    with h5py.File(data_path, 'r') as src:
        airfoil_landmark = None
        if mask_airfoil:
            airfoil_landmark = src['shape'][landmark_key][src_idx]

        out = {}
        for alpha in alphas:
            ff_group = src[alpha]['flow_field'][key]
            flow_dict = {name: ff_group[name][()] for name in ['x', 'y', *metrics]}
            out[alpha] = flow_field_to_grid(
                flow_dict,
                resolution=resolution,
                xlim=xlim,
                ylim=ylim,
                metrics=metrics,
                method=method,
                fill_nearest=fill_nearest,
                airfoil_landmark=airfoil_landmark,
                airfoil_fill_value=airfoil_fill_value,
                dtype=dtype,
            )

    return out_i, out


def flow_fields_to_grid(data_path, save_path, indices=None, alphas=('alpha+04', 'alpha+12'),
                        resolution=128, xlim=(-0.25, 1.25), ylim=(-0.5, 0.5),
                        metrics=('rho', 'rho_u', 'rho_v', 'e', 'omega'),
                        method='linear', fill_nearest=True, dtype=np.float32,
                        mask_airfoil=True, landmark_key='landmarks',
                        airfoil_fill_value=np.nan, compression='gzip',
                        progress_every=50, n_jobs=1):
    """
    Rasterise scattered HDF5 flow fields into fixed grids and save to an .h5 file.

    The output file is organised as:
      /alpha+04/rho     (N, H, W)
      /alpha+04/rho_u   (N, H, W)
      ...
      /alpha+12/rho     (N, H, W)
      /indices          selected source airfoil indices
      /x, /y            grid coordinate vectors

    Parameters
    ----------
    data_path       : source airfoil .h5 file
    save_path       : destination .h5 file
    indices         : source airfoil indices to process; None means all available
    alphas          : angle-of-attack groups to process
    resolution      : int or (height, width) output grid size
    xlim, ylim      : flow-field bounds to rasterise
    metrics         : flow variables to rasterise
    method          : scipy griddata method: 'linear', 'nearest', or 'cubic'
    fill_nearest    : fill NaNs from linear/cubic interpolation with nearest values
    dtype           : output dataset dtype
    mask_airfoil    : blank cells inside the airfoil landmark polygon
    landmark_key    : shape dataset used for the airfoil polygon mask
    airfoil_fill_value : value assigned inside the airfoil, default NaN
    compression     : HDF5 compression, or None
    progress_every  : print progress every N airfoils; set None to silence
    n_jobs          : number of worker processes; 1 keeps the original serial path.
                      Use -1 for max(1, os.cpu_count() - 1).

    Returns
    -------
    save_path
    """
    if isinstance(resolution, int):
        height = width = resolution
    else:
        height, width = resolution

    if n_jobs == -1:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = int(n_jobs)

    xs = np.linspace(xlim[0], xlim[1], width)
    ys = np.linspace(ylim[0], ylim[1], height)

    with h5py.File(data_path, 'r') as src:
        if indices is None:
            first_alpha = alphas[0]
            keys = sorted(src[first_alpha]['flow_field'].keys())
            indices = [int(key) for key in keys]
        else:
            indices = [int(idx) for idx in indices]

        with h5py.File(save_path, 'w') as dst:
            dst.create_dataset('indices', data=np.asarray(indices, dtype=np.int32))
            dst.create_dataset('x', data=xs.astype(dtype))
            dst.create_dataset('y', data=ys.astype(dtype))
            dst.attrs['source_path'] = str(data_path)
            dst.attrs['xlim'] = xlim
            dst.attrs['ylim'] = ylim
            dst.attrs['method'] = method
            dst.attrs['fill_nearest'] = fill_nearest
            dst.attrs['mask_airfoil'] = mask_airfoil
            dst.attrs['landmark_key'] = landmark_key
            dst.attrs['n_jobs'] = n_jobs

            datasets = {}
            for alpha in alphas:
                group = dst.require_group(alpha)
                datasets[alpha] = {}
                for metric in metrics:
                    datasets[alpha][metric] = group.create_dataset(
                        metric,
                        shape=(len(indices), height, width),
                        dtype=dtype,
                        compression=compression,
                    )

            if n_jobs == 1:
                for out_i, src_idx in enumerate(indices):
                    _, out = _flow_field_grid_worker((
                        data_path, out_i, src_idx, alphas, (height, width),
                        xlim, ylim, metrics, method, fill_nearest, mask_airfoil,
                        landmark_key, airfoil_fill_value, dtype
                    ))
                    for alpha, grids in out.items():
                        for metric, grid in grids.items():
                            datasets[alpha][metric][out_i] = grid

                    if progress_every and (out_i + 1) % progress_every == 0:
                        print(f'Rasterised {out_i + 1}/{len(indices)} flow fields')
            else:
                tasks = [
                    (
                        data_path, out_i, src_idx, alphas, (height, width),
                        xlim, ylim, metrics, method, fill_nearest, mask_airfoil,
                        landmark_key, airfoil_fill_value, dtype
                    )
                    for out_i, src_idx in enumerate(indices)
                ]
                completed = 0
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(_flow_field_grid_worker, task) for task in tasks]
                    for future in as_completed(futures):
                        out_i, out = future.result()
                        for alpha, grids in out.items():
                            for metric, grid in grids.items():
                                datasets[alpha][metric][out_i] = grid

                        completed += 1
                        if progress_every and completed % progress_every == 0:
                            print(f'Rasterised {completed}/{len(indices)} flow fields')

    print(f'Saved {len(indices)} flow fields x {len(alphas)} AOAs to {save_path}')
    return save_path


def plot_airfoil_grid(grids_dict, idx=0):
    """
    Plot one airfoil at all three AOAs (0, 4, 12 deg) side by side.

    grids_dict : dict returned by landmarks_to_grid (keys: alpha0, alpha4, alpha12)
    idx        : airfoil index to plot
    """
    _, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, (key, label) in zip(axes, [('alpha0', '0°'), ('alpha4', '4°'), ('alpha12', '12°')]):
        ax.imshow(grids_dict[key][idx], origin='lower', cmap='gray', interpolation='nearest')
        ax.set_title(f'AOA {label}')
        ax.axis('off')
    plt.suptitle(f'Airfoil {idx}')
    plt.tight_layout()
    plt.show()


# ── Plotting ─────────────────────────────────────────────────────────────────────

def plot_airfoil_geometries(landmarks, n=30, title='Sample Airfoil Geometries'):
    """Plot the first n airfoil profiles from a landmarks array."""
    plt.figure(figsize=(10, 6))
    for i, lm in enumerate(landmarks[:n]):
        plt.plot(lm[:, 0], lm[:, 1], label=f'airfoil {i}')
    plt.xlabel('x/c', fontsize=12)
    plt.ylabel('y/c', fontsize=12)
    plt.gca().set_aspect(1.)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _param_limits(data):
    """Compute rounded (lower, upper) limits for each column of data."""
    limits = []
    for i in range(data.shape[1]):
        lo, hi = data[:, i].min(), data[:, i].max()
        lo = rounders.floor(lo, abs(int(math.log10(abs(lo)))) + 1)
        hi = rounders.ceil(hi, abs(int(math.log10(abs(hi)))) + 1)
        limits.append([lo, hi])
    return limits


def plot_pairwise_distributions(data, labels, figsize=None):
    """
    Lower-triangle pairwise scatter + diagonal KDE plot.

    data:   (N, m) numpy array
    labels: list of m strings
    """
    m = data.shape[1]
    figsize = figsize or (max(7, m * 1.8), max(7, m * 1.8))
    limits = _param_limits(data)

    plt.figure(figsize=figsize)
    for i in range(m):
        for j in range(i + 1):
            ax = plt.subplot(m, m, m * i + j + 1)
            lim_i, lim_j = limits[i], limits[j]

            if i == j:
                vals = np.linspace(data[:, i].min(), data[:, i].max())
                kde = gaussian_kde(data[:, i])
                plt.plot(vals, kde(vals), 'k')
                plt.xlim(lim_i[0] - 0.05 * np.ptp(lim_i), lim_i[-1] + 0.05 * np.ptp(lim_i))
                plt.yticks([])
                plt.ylabel('Density')
            else:
                plt.scatter(data[:, j], data[:, i], s=40, c=[[0.4, 0.4, 0.4]], edgecolor='k')
                plt.xticks(lim_j)
                plt.yticks(lim_i)
                plt.xlim(lim_j[0] - 0.05 * np.ptp(lim_j), lim_j[-1] + 0.05 * np.ptp(lim_j))
                plt.ylim(lim_i[0] - 0.05 * np.ptp(lim_i), lim_i[-1] + 0.05 * np.ptp(lim_i))
                if j == 0:
                    plt.ylabel(labels[i], fontsize=10)
                else:
                    plt.yticks([])

            if i == m - 1:
                plt.xlabel(labels[j], fontsize=10)
            else:
                plt.xticks([])

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

    plt.tight_layout()
    plt.show()


def plot_aero_violins(cl04, cd04, cm04, cl12, cd12, cm12):
    """Violin plots of C_l, C_d, C_m for both angles of attack."""
    plt.figure(figsize=(12, 4))
    for i, (d04, d12, ylabel) in enumerate([
        (cl04, cl12, 'Coefficient of Lift'),
        (cd04, cd12, 'Coefficient of Drag'),
        (cm04, cm12, 'Coefficient of Pitching Moment'),
    ], 1):
        plt.subplot(1, 3, i)
        ax = sns.violinplot(data=[d04, d12])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([4, 12])
        plt.xlabel('Angle of Attack (deg.)')
        plt.ylabel(ylabel)
    plt.suptitle('Distribution of Aerodynamic Coefficients by AOA')
    plt.tight_layout()
    plt.show()


def plot_aero_vs_geometry(geom_values, cl04, cd04, cm04, cl12, cd12, cm12, xlabel='Geometry'):
    """Scatter plots of C_l, C_d, C_m vs a geometry scalar for both AOAs."""
    plt.figure(figsize=(12, 4))
    for i, (d04, d12, ylabel) in enumerate([
        (cl04, cl12, 'Coefficient of Lift'),
        (cd04, cd12, 'Coefficient of Drag'),
        (cm04, cm12, 'Coefficient of Pitching Moment'),
    ], 1):
        plt.subplot(1, 3, i)
        plt.scatter(geom_values, d04, c='tab:orange', label='alpha+04', s=5)
        plt.scatter(geom_values, d12, c='tab:blue', label='alpha+12', s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if i == 3:
            plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_flow_field_plotly(flow_dict, idx, metric='rho', xlim=(-0.25, 1.25), ylim=(-0.5, 0.5)):
    """
    Plotly scatter plot for a single flow-field metric.

    metric: one of 'rho', 'rho_u', 'rho_v', 'e', 'omega'

    Usage:
        ff = load_flow_field(data_path, idx=42)
        plot_flow_field_plotly(ff, idx=42, metric='rho_u')
    """
    import plotly.graph_objects as go

    valid = ['rho', 'rho_u', 'rho_v', 'e', 'omega']
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}")

    x, y = flow_dict['x'], flow_dict['y']
    tf = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])

    val = flow_dict[metric][tf]

    fig = go.Figure(go.Scatter(
        x=x[tf], y=y[tf],
        mode='markers',
        marker=dict(color=val, colorscale='Viridis', size=2,
                    colorbar=dict(title=metric), showscale=True),
    ))
    fig.update_layout(
        title=f'Airfoil {idx:04d} — {metric}',
        xaxis=dict(range=list(xlim), title='x/c'),
        yaxis=dict(range=list(ylim), title='y/c'),
        width=800, height=500,
    )
    fig.show()


def plot_flow_field(flow_dict, idx, xlim=(-0.25, 1.25), ylim=(-0.5, 0.5)):
    """Scatter-colour plots of all flow-field variables for one airfoil."""
    val_list = ['rho', 'rho_u', 'rho_v', 'e', 'omega']
    x, y = flow_dict['x'], flow_dict['y']
    tf = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])

    plt.figure(figsize=(12, 5))
    for i, v in enumerate(val_list):
        val = flow_dict[v]
        vmin, vmax = val[tf].min(), val[tf].max()
        plt.subplot(2, 3, i + 1)
        plt.scatter(x, y, c=val, vmin=vmin, vmax=vmax, s=3)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.colorbar()
        plt.title(v)
        plt.gca().set_aspect(1.)

    plt.suptitle(f'Airfoil {idx:04d}')
    plt.tight_layout()
    plt.show()


# ── Model Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, x_test, y_test, labels=('cl', 'cd', 'cm')):
    """
    Print per-output regression metrics and plot predicted-vs-actual + residuals.

    Parameters
    ----------
    model  : trained Keras model
    x_test : (N, 128, 128, 1) float32 array
    y_test : (N, 3) float32 array
    labels : coefficient names, default ('cl', 'cd', 'cm')
    """
    labels = list(labels)

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")

    y_pred = model.predict(x_test, verbose=0)
    y_true = y_test

    print("\nPer-output metrics:")
    for i, name in enumerate(labels):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        print(f"  {name}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Predicted vs Actual
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (ax, name) in enumerate(zip(axes, labels)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, edgecolors='k', linewidths=0.3)
        mn, mx = y_true[:, i].min(), y_true[:, i].max()
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name} — R²={r2_score(y_true[:, i], y_pred[:, i]):.3f}")
    plt.suptitle("Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    # Residuals
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, name in enumerate(labels):
        residuals = y_true[:, i] - y_pred[:, i]

        axes[0, i].scatter(y_pred[:, i], residuals, alpha=0.7, edgecolors='k', linewidths=0.3)
        axes[0, i].axhline(0, color='r', linestyle='--', linewidth=1)
        axes[0, i].set_xlabel(f"Predicted {name}")
        axes[0, i].set_ylabel("Residual")
        axes[0, i].set_title(f"{name} residuals")

        axes[1, i].hist(residuals, bins=7, edgecolor='k')
        axes[1, i].axvline(0, color='r', linestyle='--', linewidth=1)
        axes[1, i].set_xlabel("Residual")
        axes[1, i].set_ylabel("Count")
        axes[1, i].set_title(f"{name} residual distribution")

    plt.suptitle("Residual Plots")
    plt.tight_layout()
    plt.show()

def plot_training_sample(
    X_train,
    Y_train,
    M_train=None,
    idx=0,
    normalized_y=False,
    Y_mean=None,
    Y_std=None
):
    x_sample = X_train[idx]
    y_sample = Y_train[idx].copy()

    if y_sample.shape[-1] > 4:
        y_sample = y_sample[..., :4]

    if normalized_y:
        if Y_mean is None or Y_std is None:
            raise ValueError("Pass Y_mean and Y_std when normalized_y=True.")
        y_sample = y_sample * Y_std.reshape(1, 1, -1) + Y_mean.reshape(1, 1, -1)

    if M_train is not None:
        mask = M_train[idx]
        if mask.ndim == 3:
            mask = mask[..., 0]

        y_plot = y_sample.copy()
        for c in range(y_plot.shape[-1]):
            y_plot[..., c] = np.where(mask == 1, y_plot[..., c], np.nan)
    else:
        y_plot = y_sample

    airfoil = x_sample[..., 0]

    if x_sample.shape[-1] > 1:
        aoa_channel = x_sample[..., 1]
    else:
        aoa_channel = None

    field_names = ["rho", "rho_u", "rho_v", "e"]

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    plt.figure(figsize=(14, 8))

    # Airfoil input
    plt.subplot(2, 3, 1)
    plt.imshow(
        airfoil,
        cmap="gray",
        origin="lower",
        interpolation="nearest",
        aspect="equal"
    )
    plt.title("Input: Airfoil Raster")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    # AoA channel
    plt.subplot(2, 3, 2)
    if aoa_channel is not None:
        plt.imshow(
            aoa_channel,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            aspect="equal"
        )
        aoa_value_norm = np.nanmean(aoa_channel)
        aoa_deg = aoa_value_norm * 4 + 8
        plt.title(f"Input: AoA Channel ≈ {aoa_deg:.1f}°")
        plt.colorbar(fraction=0.046, pad=0.04)
    else:
        plt.text(0.5, 0.5, "No AoA channel", ha="center", va="center")
        plt.title("AoA Channel")
    plt.axis("off")

    # Flow fields
    for i, name in enumerate(field_names):
        plt.subplot(2, 3, i + 3)
        plt.imshow(
            np.ma.masked_invalid(y_plot[..., i]),
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            aspect="equal"
        )
        plt.title(f"Target: {name}")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# FOR MODIFYING FLOW FIELDS BEFORE MODEL TRAINING
def fill_nan_nearest(arr):
    """
    Fill NaNs using nearest finite values.
    This is used before rotation so interpolation does not spread NaNs.
    """
    arr = np.asarray(arr)

    valid = np.isfinite(arr)

    if valid.all():
        return arr.astype(np.float32)

    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)

    nearest_idx = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True
    )

    filled = arr[tuple(nearest_idx)]

    return filled.astype(np.float32)


def rotate_scalar_field_keep_solid_nan(field, solid_mask, angle_deg, order=1):
    """
    Rotate scalar field while:
    - filling exterior white/corner gaps with nearest values
    - restoring the rotated airfoil/solid region to NaN
    """

    field_filled = fill_nan_nearest(field)

    rotated_field = rotate(
        field_filled,
        angle=angle_deg,
        reshape=False,
        order=order,
        mode="nearest"
    ).astype(np.float32)

    rotated_solid_mask = rotate(
        solid_mask.astype(np.float32),
        angle=angle_deg,
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0
    ) > 0.5

    rotated_field[rotated_solid_mask] = np.nan

    return rotated_field, rotated_solid_mask


def rotate_one_case(
    rho,
    rho_u,
    rho_v,
    e,
    angle_deg,
    omega=None,
    rotate_vector_components=True
    ):
    """
    Rotates one airfoil CFD case.

    rho, e, omega are scalar-like fields.
    rho_u and rho_v are vector/momentum components.

    If rotate_vector_components=True:
        spatial image is rotated first,
        then rho_u/rho_v components are rotated as a vector.
    """

    # Solid airfoil region from NaNs
    solid_mask = ~np.isfinite(rho)

    # Rotate scalar rho
    rho_rot, solid_rot = rotate_scalar_field_keep_solid_nan(
        rho,
        solid_mask,
        angle_deg=angle_deg,
        order=1
    )

    # Rotate momentum-component images spatially
    rho_u_img_rot, _ = rotate_scalar_field_keep_solid_nan(
        rho_u,
        solid_mask,
        angle_deg=angle_deg,
        order=1
    )

    rho_v_img_rot, _ = rotate_scalar_field_keep_solid_nan(
        rho_v,
        solid_mask,
        angle_deg=angle_deg,
        order=1
    )

    # Rotate energy scalar
    e_rot, _ = rotate_scalar_field_keep_solid_nan(
        e,
        solid_mask,
        angle_deg=angle_deg,
        order=1
    )

    # Rotate vector components after spatial rotation
    if rotate_vector_components:
        theta = np.deg2rad(angle_deg)

        rho_u_rot = rho_u_img_rot * np.cos(theta) - rho_v_img_rot * np.sin(theta)
        rho_v_rot = rho_u_img_rot * np.sin(theta) + rho_v_img_rot * np.cos(theta)

        rho_u_rot[solid_rot] = np.nan
        rho_v_rot[solid_rot] = np.nan
    else:
        rho_u_rot = rho_u_img_rot
        rho_v_rot = rho_v_img_rot

    outputs = {
        "rho": rho_rot.astype(np.float32),
        "rho_u": rho_u_rot.astype(np.float32),
        "rho_v": rho_v_rot.astype(np.float32),
        "e": e_rot.astype(np.float32),
        "solid_mask": solid_rot.astype(np.float32),
    }

    if omega is not None:
        omega_rot, _ = rotate_scalar_field_keep_solid_nan(
            omega,
            solid_mask,
            angle_deg=angle_deg,
            order=1
        )
        outputs["omega"] = omega_rot.astype(np.float32)

    return outputs


def build_rotated_flow_dataset(
    data_flow,
    save_path,
    rotation_sign=1,
    include_omega=True,
    compression="gzip"
    ):
    """
    Creates a new dataset from already-rasterized flow fields.

    Inputs are extracted from rotated NaN regions:
        X[..., 0] = solid airfoil mask
        X[..., 1] = normalized AoA channel

    Targets:
        Y[..., 0] = rho
        Y[..., 1] = rho_u
        Y[..., 2] = rho_v
        Y[..., 3] = e

    Also saves:
        fluid_mask
        solid_mask
        alpha_deg
        x, y coordinate vectors
    """

    alpha_specs = [
        {
            "group": "alpha+04",
            "alpha_deg": 4.0,
        },
        {
            "group": "alpha+12",
            "alpha_deg": 12.0,
        },
    ]

    all_X = []
    all_Y = []
    all_solid_masks = []
    all_fluid_masks = []
    all_alpha_deg = []

    all_omega = [] if include_omega else None

    with h5py.File(data_flow, "r") as hf:
        x = hf["x"][()]
        y = hf["y"][()]

        for spec in alpha_specs:
            group = spec["group"]
            alpha_deg = spec["alpha_deg"]

            angle_to_rotate = rotation_sign * alpha_deg

            rho_all = hf[group]["rho"][()]
            rho_u_all = hf[group]["rho_u"][()]
            rho_v_all = hf[group]["rho_v"][()]
            e_all = hf[group]["e"][()]

            if include_omega and "omega" in hf[group]:
                omega_all = hf[group]["omega"][()]
            else:
                omega_all = None

            n = rho_all.shape[0]

            print(f"Processing {group}: {n} cases, rotation={angle_to_rotate} degrees")

            for i in range(n):
                omega_i = omega_all[i] if omega_all is not None else None

                rotated = rotate_one_case(
                    rho=rho_all[i],
                    rho_u=rho_u_all[i],
                    rho_v=rho_v_all[i],
                    e=e_all[i],
                    omega=omega_i,
                    angle_deg=angle_to_rotate,
                    rotate_vector_components=True
                )

                solid_mask = rotated["solid_mask"][..., np.newaxis]
                fluid_mask = 1.0 - solid_mask

                # Normalized AoA:
                # 4 deg -> -1
                # 12 deg -> +1
                alpha_norm = (alpha_deg - 8.0) / 4.0
                alpha_channel = np.ones_like(solid_mask, dtype=np.float32) * alpha_norm

                X_i = np.concatenate(
                    [solid_mask, alpha_channel],
                    axis=-1
                ).astype(np.float32)

                Y_i = np.stack(
                    [
                        rotated["rho"],
                        rotated["rho_u"],
                        rotated["rho_v"],
                        rotated["e"],
                    ],
                    axis=-1
                ).astype(np.float32)

                all_X.append(X_i)
                all_Y.append(Y_i)
                all_solid_masks.append(solid_mask.astype(np.float32))
                all_fluid_masks.append(fluid_mask.astype(np.float32))
                all_alpha_deg.append(alpha_deg)

                if include_omega and "omega" in rotated:
                    all_omega.append(rotated["omega"].astype(np.float32))

    X = np.stack(all_X, axis=0).astype(np.float32)
    Y = np.stack(all_Y, axis=0).astype(np.float32)
    solid_mask = np.stack(all_solid_masks, axis=0).astype(np.float32)
    fluid_mask = np.stack(all_fluid_masks, axis=0).astype(np.float32)
    alpha_deg = np.asarray(all_alpha_deg, dtype=np.float32)

    print("Final X:", X.shape)
    print("Final Y:", Y.shape)
    print("solid_mask:", solid_mask.shape)
    print("fluid_mask:", fluid_mask.shape)
    print("alpha_deg:", alpha_deg.shape)

    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("X", data=X, compression=compression)
        hf.create_dataset("Y", data=Y, compression=compression)

        hf.create_dataset("solid_mask", data=solid_mask, compression=compression)
        hf.create_dataset("fluid_mask", data=fluid_mask, compression=compression)
        hf.create_dataset("alpha_deg", data=alpha_deg, compression=compression)

        hf.create_dataset("x", data=x.astype(np.float32))
        hf.create_dataset("y", data=y.astype(np.float32))

        hf.attrs["description"] = (
            "Rotated flow-field dataset. "
            "X[...,0]=solid airfoil mask from rotated NaNs, "
            "X[...,1]=normalized AoA channel. "
            "Y[...,0:4]=rho,rho_u,rho_v,e."
        )
        hf.attrs["rotation_sign"] = rotation_sign
        hf.attrs["xlim"] = (float(x.min()), float(x.max()))
        hf.attrs["ylim"] = (float(y.min()), float(y.max()))
        hf.attrs["target_fields"] = "rho,rho_u,rho_v,e"
        hf.attrs["input_fields"] = "solid_mask,alpha_norm"
        hf.attrs["alpha_norm_formula"] = "(alpha_deg - 8.0) / 4.0"
        hf.attrs["vector_components_rotated"] = True

        if include_omega and all_omega is not None and len(all_omega) > 0:
            omega = np.stack(all_omega, axis=0).astype(np.float32)
            hf.create_dataset("omega", data=omega, compression=compression)
            hf.attrs["omega_included"] = True
        else:
            hf.attrs["omega_included"] = False

    print(f"Saved rotated dataset to: {save_path}")

    return save_path

def plot_rotated_dataset_sample_with_rho_preview(
    path="data/airfoil_1k_rotated_nanmask_inputs_128.h5",
    idx=0,
    rho_scaled=True,
    ):
    with h5py.File(path, "r") as hf:
        X = hf["X"][idx]        # channels: solid_mask, alpha_norm
        Y = hf["Y"][idx]        # channels: rho, rho_u, rho_v, e
        alpha_deg = hf["alpha_deg"][idx]
        x = hf["x"][()]
        y = hf["y"][()]

    solid_mask = X[:, :, 0]

    rho = Y[:, :, 0]
    rho_u = Y[:, :, 1]
    rho_v = Y[:, :, 2]
    e = Y[:, :, 3]

    # 2nd subplot only: scaled rho preview in pixel space
    rho_preview = rho.copy()

    if rho_scaled:
        valid = np.isfinite(rho_preview)
        rho_min = np.nanmin(rho_preview)
        rho_max = np.nanmax(rho_preview)

        rho_preview = (rho_preview - rho_min) / (rho_max - rho_min + 1e-8)
        rho_preview[~valid] = np.nan

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    field_names = ["rho", "rho_u", "rho_v", "e"]
    fields = [rho, rho_u, rho_v, e]

    plt.figure(figsize=(14, 8))

    # 1. Input solid mask — pixel space
    plt.subplot(2, 3, 1)
    plt.imshow(
        solid_mask,
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        aspect="equal",
    )
    plt.title(f"Input: Solid Mask | alpha={alpha_deg:.0f}°")
    plt.axis("off")

    # 2. Rho preview — pixel space, same shape as solid mask
    plt.subplot(2, 3, 2)
    plt.imshow(
        np.ma.masked_invalid(rho_preview),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    plt.title("Preview: rho scaled 0–1")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    # 3–6. Original output fields — keep physical extent unchanged
    for i, (name, arr) in enumerate(zip(field_names, fields)):
        plt.subplot(2, 3, i + 3)
        plt.imshow(
            np.ma.masked_invalid(arr),
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        plt.title(f"Output: {name}")
        plt.xlabel("x/c")
        plt.ylabel("y/c")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# Viewing the Saved UNet Model ===========================================================
def plot_saved_model_prediction(
    model_path,
    X_data,
    Y_true_raw,
    M_data,
    Y_mean,
    Y_std,
    idx=0,
    channel=0,
    field_names=("rho", "rho_u", "rho_v", "e"),
    percentile_clip=(1, 99),
):
    """
    Loads a saved Keras U-Net model and plots:
    - True flow field
    - Predicted flow field
    - Absolute error

    Parameters
    ----------
    model_path : str
        Path to saved Keras model, e.g. "models/airfoil_unet_all_fields.keras"

    X_data : ndarray
        Model inputs, shape (N, 128, 128, 2)

    Y_true_raw : ndarray
        Unnormalized physical target fields, shape (N, 128, 128, 4)

    M_data : ndarray
        Fluid mask, shape (N, 128, 128, 1)
        1 = fluid region, 0 = airfoil/solid region

    Y_mean, Y_std : ndarray
        Per-channel normalization values used during training.
        Shape should be (1, 1, 1, 4)

    idx : int
        Sample index to plot.

    channel : int
        0 = rho
        1 = rho_u
        2 = rho_v
        3 = e
    """

    # Load saved model.
    # compile=False avoids needing to reload the custom masked_mse loss.
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Predict one sample
    X_sample = X_data[idx:idx + 1]
    Y_pred_norm = model.predict(X_sample, verbose=0)

    # Unnormalize prediction
    Y_pred = Y_pred_norm * Y_std + Y_mean

    # Get true and predicted field
    true_plot = Y_true_raw[idx, :, :, channel].copy()
    pred_plot = Y_pred[0, :, :, channel].copy()

    # Restore solid/airfoil region as NaN
    mask = M_data[idx, :, :, 0]

    true_plot[mask == 0] = np.nan
    pred_plot[mask == 0] = np.nan

    error_plot = np.abs(true_plot - pred_plot)
    error_plot[mask == 0] = np.nan

    # Color scale based on true field
    vmin = np.nanpercentile(true_plot, percentile_clip[0])
    vmax = np.nanpercentile(true_plot, percentile_clip[1])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(
        np.ma.masked_invalid(true_plot),
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"True {field_names[channel]}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 2)
    plt.imshow(
        np.ma.masked_invalid(pred_plot),
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Predicted {field_names[channel]}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(
        np.ma.masked_invalid(error_plot),
        cmap="magma",
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )
    plt.title("Absolute Error")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "Y_pred": Y_pred,
        "true_field": true_plot,
        "pred_field": pred_plot,
        "error_field": error_plot,
    }