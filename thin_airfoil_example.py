from raster_airfoil import orient_airfoil_left_to_right, raster_to_xy
from thin_airfoil import erau_modified_thin_airfoil


def compare_raster_airfoil_to_dataset(
    x_data,
    cl04,
    cd04,
    cm04,
    cl12,
    cd12,
    cm12,
    idx=5,
    mach=0.1,
    reynolds=9_000_000,
    flip_y=True,
    flip_x=False,
):
    """
    Mirror the notebook thin-airfoil comparison for one rasterized airfoil.

    The data arrays are passed in explicitly so this file can be imported without
    relying on notebook globals.
    """
    xy = raster_to_xy(x_data[idx])
    xy_lr = orient_airfoil_left_to_right(xy, flip_y=flip_y, flip_x=flip_x)

    result_4deg = erau_modified_thin_airfoil(
        xy_lr,
        alpha_deg=4,
        mach=mach,
        reynolds=reynolds,
        flip_y=False,
    )
    result_12deg = erau_modified_thin_airfoil(
        xy_lr,
        alpha_deg=12,
        mach=mach,
        reynolds=reynolds,
        flip_y=False,
    )

    return {
        "xy": xy,
        "xy_oriented": xy_lr,
        "thin_airfoil": {
            "4deg": result_4deg,
            "12deg": result_12deg,
        },
        "dataset": {
            "4deg": {
                "Cl": cl04[idx],
                "Cd": cd04[idx],
                "Cm": cm04[idx],
            },
            "12deg": {
                "Cl": cl12[idx],
                "Cd": cd12[idx],
                "Cm": cm12[idx],
            },
        },
    }
