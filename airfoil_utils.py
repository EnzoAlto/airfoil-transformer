import h5py
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
import math
import rounders
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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
