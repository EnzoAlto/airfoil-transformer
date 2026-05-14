import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d


def erau_modified_thin_airfoil(
    xy,
    alpha_deg,
    mach=0.1,
    reynolds=9_000_000,
    eta_a=0.95,
    cd0=None,
    flip_y=False,
    n_bins=500,
    n_theta=800,
    use_prandtl_glauert=True,
):
    """
    Modified thin airfoil theory with an empirical drag add-on.

    Designed for unordered 2D airfoil contour XY points.
    Returns Cl, Cd, Cm_c4, alpha_zero_lift_deg, A0, A1, and A2,
    along with intermediate diagnostic values.
    """
    xy = np.asarray(xy, dtype=float)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")

    x = xy[:, 0].copy()
    y = xy[:, 1].copy()

    if flip_y:
        y = -y

    x_min, x_max = np.min(x), np.max(x)
    chord = x_max - x_min

    if chord <= 0:
        raise ValueError("Invalid chord length.")

    x_bar = (x - x_min) / chord
    y_bar = y / chord

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.digitize(x_bar, bins) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    x_mid = []
    y_upper = []
    y_lower = []

    for i in range(n_bins):
        mask = bin_id == i
        if np.sum(mask) < 1:
            continue

        xs = x_bar[mask]
        ys = y_bar[mask]

        x_mid.append(np.mean(xs))
        y_upper.append(np.max(ys))
        y_lower.append(np.min(ys))

    x_mid = np.asarray(x_mid)
    y_upper = np.asarray(y_upper)
    y_lower = np.asarray(y_lower)

    order = np.argsort(x_mid)
    x_mid = x_mid[order]
    y_upper = y_upper[order]
    y_lower = y_lower[order]

    _, unique_idx = np.unique(x_mid, return_index=True)
    x_mid = x_mid[unique_idx]
    y_upper = y_upper[unique_idx]
    y_lower = y_lower[unique_idx]

    xg = np.linspace(np.min(x_mid), np.max(x_mid), n_bins)

    upper_interp = interp1d(
        x_mid, y_upper, kind="linear", fill_value="extrapolate"
    )
    lower_interp = interp1d(
        x_mid, y_lower, kind="linear", fill_value="extrapolate"
    )

    yu = upper_interp(xg)
    yl = lower_interp(xg)

    yc = 0.5 * (yu + yl)
    dyc_dx = np.gradient(yc, xg)

    theta = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    x_theta = 0.5 * (1.0 - np.cos(theta))

    dyc_theta = interp1d(
        xg, dyc_dx, kind="linear", fill_value="extrapolate"
    )(x_theta)

    alpha = np.deg2rad(alpha_deg)

    i0 = (1.0 / np.pi) * simpson(dyc_theta, x=theta)
    a0 = alpha - i0
    a1 = (2.0 / np.pi) * simpson(dyc_theta * np.cos(theta), x=theta)
    a2 = (2.0 / np.pi) * simpson(dyc_theta * np.cos(2.0 * theta), x=theta)

    cl = 2.0 * np.pi * (a0 + 0.5 * a1)
    cm_c4 = (np.pi / 4.0) * (a2 - a1)
    alpha_zero_lift = i0 - 0.5 * a1

    if cd0 is None:
        cf = 0.074 / (reynolds ** 0.2)
        upper_length = np.sum(np.sqrt(np.diff(xg) ** 2 + np.diff(yu) ** 2))
        lower_length = np.sum(np.sqrt(np.diff(xg) ** 2 + np.diff(yl) ** 2))
        wetted_ratio = upper_length + lower_length
        cd0_used = cf * wetted_ratio
    else:
        cd0_used = float(cd0)
        wetted_ratio = np.nan

    cd_pressure = ((1.0 - eta_a) / (2.0 * np.pi)) * cl**2
    cd = cd0_used + cd_pressure

    if use_prandtl_glauert:
        beta = np.sqrt(1.0 - mach**2)
        cl = cl / beta
        cm_c4 = cm_c4 / beta

    thickness = yu - yl
    max_thickness = np.max(thickness)

    return {
        "Cl": cl,
        "Cd": cd,
        "Cm_c4": cm_c4,
        "alpha_zero_lift_deg": np.rad2deg(alpha_zero_lift),
        "A0": a0,
        "A1": a1,
        "A2": a2,
        "cd0_used": cd0_used,
        "Cd_pressure_component": cd_pressure,
        "eta_a": eta_a,
        "wetted_ratio": wetted_ratio,
        "max_thickness_to_chord": max_thickness,
        "mach": mach,
        "reynolds": reynolds,
        "alpha_deg": alpha_deg,
        "flip_y": flip_y,
    }
