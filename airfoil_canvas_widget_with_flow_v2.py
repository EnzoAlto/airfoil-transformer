import base64
import io

import numpy as np
from PIL import Image, ImageFilter
from IPython.display import HTML, clear_output, display
import ipywidgets as widgets
from ipycanvas import Canvas

try:
    from scipy.ndimage import binary_fill_holes
except Exception:  # pragma: no cover
    binary_fill_holes = None

import matplotlib.pyplot as plt


CANVAS_W, CANVAS_H = 280, 280
MODEL_W, MODEL_H = 128, 128
LABELS = ['Cl', 'Cd', 'Cm']
FLOW_FIELD_NAMES = ('rho', 'rho_u', 'rho_v', 'e')

# Module-level singleton: reuse the same Output widget across re-runs so that
# re-executing the cell does not stack multiple live Output widgets in the cell.
_WIDGET_OUTPUT = None


def calculate_angle_of_attack(image_data, threshold=25, min_pixels=20):
    """
    Estimate angle of attack from a drawn airfoil image.

    The estimate fits the dominant left-to-right direction of the white pixels.
    Positive angles mean the leading edge points upward on the displayed canvas.
    """
    image_data = np.asarray(image_data)
    if image_data.ndim == 3:
        gray = image_data[:, :, 0]
    else:
        gray = image_data

    points_yx = np.argwhere(gray > threshold)
    if len(points_yx) < min_pixels:
        return None

    x = points_yx[:, 1].astype(np.float32)
    y = points_yx[:, 0].astype(np.float32)
    points = np.column_stack([x, y])
    points -= points.mean(axis=0, keepdims=True)

    _, _, vh = np.linalg.svd(points, full_matrices=False)
    vx, vy = vh[0]
    if vx < 0:
        vx, vy = -vx, -vy

    angle_deg = np.degrees(np.arctan2(-vy, vx))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return float(angle_deg)


def _mc_predict(model, x, n_mc):
    """Run Monte Carlo dropout predictions and return mean/std per output."""
    runs = np.stack(
        [model(x, training=True).numpy() for _ in range(n_mc)],
        axis=0,
    )
    return runs.mean(axis=0)[0], runs.std(axis=0)[0]


def _resize_gray_to_model(gray):
    """Resize canvas grayscale image to the model grid."""
    img = Image.fromarray(gray, mode='L').resize((MODEL_W, MODEL_H), Image.LANCZOS)
    return np.array(img).astype('float32') / 255.0


def _smooth_gray_for_flow_preview(gray, radius=1.25):
    """
    Smooth the drawn canvas image only for the U-Net mask/flow preview path.

    This does not alter the canvas drawing and does not affect the coefficient
    prediction model.
    """
    radius = float(radius)
    if radius <= 0:
        return np.asarray(gray, dtype=np.uint8)

    img = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode='L')
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img, dtype=np.uint8)


def _make_solid_mask_from_gray(gray_model, threshold=0.10, fill_holes=True):
    """
    Convert resized gray canvas image into a solid-mask input for the U-Net.

    Returns mask with shape (128, 128), where:
      1 = solid airfoil
      0 = fluid/background
    """
    mask = gray_model > threshold

    if fill_holes and binary_fill_holes is not None:
        # This works best when the user draws a closed airfoil outline.
        mask = binary_fill_holes(mask)

    return mask.astype('float32')


def _flow_alpha_channel(angle_deg, solid_mask, alpha_min=4.0, alpha_max=12.0, clip=True):
    """
    Make AoA channel used by the U-Net.

    Training convention:
      alpha = 4 deg  -> -1
      alpha = 12 deg -> +1
      alpha_norm = (alpha - 8) / 4
    """
    if angle_deg is None:
        alpha_used = 4.0
    else:
        alpha_used = float(angle_deg)

    if clip:
        alpha_used = float(np.clip(alpha_used, alpha_min, alpha_max))

    alpha_norm = (alpha_used - 8.0) / 4.0
    return np.ones_like(solid_mask, dtype=np.float32) * alpha_norm, alpha_used, alpha_norm


def _predict_flow_field(
    flow_model,
    gray_model,
    angle_deg,
    flow_y_mean,
    flow_y_std,
    field_channel=0,
    mask_threshold=0.10,
    fill_holes=True,
    clip_alpha=True,
    flip_y_for_model=False,
    flip_y_for_display=False,
):
    """
    Build U-Net input from the canvas and return one physical-scale flow field.
    """
    # Mask in canvas/display orientation.
    solid_mask_display = _make_solid_mask_from_gray(
        gray_model,
        threshold=mask_threshold,
        fill_holes=fill_holes,
    )

    # Mask in model/training orientation.
    solid_mask_model = np.flipud(solid_mask_display) if flip_y_for_model else solid_mask_display

    alpha_channel, alpha_used, alpha_norm = _flow_alpha_channel(
        angle_deg,
        solid_mask_model,
        clip=clip_alpha,
    )

    x_flow = np.stack([solid_mask_model, alpha_channel], axis=-1)[np.newaxis].astype('float32')
    y_pred_norm = flow_model.predict(x_flow, verbose=0)

    y_mean = np.asarray(flow_y_mean, dtype=np.float32).reshape(1, 1, 1, -1)
    y_std = np.asarray(flow_y_std, dtype=np.float32).reshape(1, 1, 1, -1)
    y_pred_model = y_pred_norm * y_std + y_mean

    field_model = y_pred_model[0, :, :, field_channel].astype(np.float32)
    field_model[solid_mask_model > 0.5] = np.nan

    # Convert prediction back to canvas/display orientation if requested.
    if flip_y_for_display:
        field_display = np.flipud(field_model)
        all_fields_display = np.flipud(y_pred_model[0])
        solid_mask_preview = solid_mask_display
        alpha_channel_preview = np.flipud(alpha_channel) if flip_y_for_model else alpha_channel
    else:
        field_display = field_model
        all_fields_display = y_pred_model[0]
        solid_mask_preview = solid_mask_model
        alpha_channel_preview = alpha_channel

    field_display[solid_mask_preview > 0.5] = np.nan

    return {
        'field': field_display,
        'all_fields': all_fields_display,
        'solid_mask': solid_mask_preview,
        'solid_mask_model': solid_mask_model,
        'alpha_channel': alpha_channel_preview,
        'alpha_used': alpha_used,
        'alpha_norm': alpha_norm,
    }


def _array_to_png_data_uri(
    arr,
    cmap_name='viridis',
    vmin=None,
    vmax=None,
    title=None,
    origin='upper',
):
    """
    Render a 2D array to a small PNG data URI for HTML display.

    origin='upper' is intentional for the widget preview so the mask/flow image
    visually matches the canvas drawing orientation.
    """
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('white')

    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=110)
    im = ax.imshow(
        masked,
        cmap=cmap,
        origin=origin,
        interpolation='nearest',
        aspect='equal',
        vmin=vmin,
        vmax=vmax,
    )
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{b64}'


def create_airfoil_draw_widget(
    model,
    scales_np,
    n_mc=50,
    flow_model=None,
    flow_y_mean=None,
    flow_y_std=None,
    flow_field_channel=0,
    flow_mask_threshold=0.10,
    flow_fill_holes=True,
    flow_clip_alpha=True,
    flow_flip_y_for_model=True,
    flow_flip_y_for_display=True,
):
    """
    Create and display a drawable airfoil prediction widget.

    Parameters
    ----------
    model : keras model
        Coefficient model that accepts input shape (1, 128, 128, 1).
    scales_np : array-like
        Output scaling values for [Cl, Cd, Cm].
    n_mc : int
        Number of MC dropout prediction passes.

    flow_model : keras model, optional
        U-Net model that accepts input shape (1, 128, 128, 2):
        channel 0 = solid mask, channel 1 = normalized AoA.
    flow_y_mean, flow_y_std : array-like, optional
        Per-channel normalization stats for [rho, rho_u, rho_v, e].
    flow_field_channel : int
        Initial flow field to display: 0=rho, 1=rho_u, 2=rho_v, 3=e.
    flow_flip_y_for_model : bool
        If True, flips the drawn mask vertically before sending it to the
        U-Net. This fixes canvas-vs-training y-axis convention mismatches.
    flow_flip_y_for_display : bool
        If True, flips the predicted field back vertically for the widget
        preview so the user can draw normally on the canvas.
    """
    global _WIDGET_OUTPUT
    scales_np = np.asarray(scales_np, dtype=np.float32)
    flow_enabled = flow_model is not None and flow_y_mean is not None and flow_y_std is not None

    canvas = Canvas(width=CANVAS_W, height=CANVAS_H, sync_image_data=True)
    conditions = widgets.HTML()
    wind_indicator = widgets.HTML()

    # Reuse the same Output widget across re-runs so old instances don't
    # accumulate in the cell and cause the output to appear multiple times.
    if _WIDGET_OUTPUT is None:
        _WIDGET_OUTPUT = widgets.Output()
    else:
        _WIDGET_OUTPUT.clear_output(wait=True)
    output = _WIDGET_OUTPUT
    btn_predict = widgets.Button(description='Predict', button_style='primary')
    btn_clear = widgets.Button(description='Clear')
    btn_up = widgets.Button(description='Up')
    btn_down = widgets.Button(description='Down')
    btn_left = widgets.Button(description='Left')
    btn_right = widgets.Button(description='Right')
    btn_rotate_ccw = widgets.Button(description='Rot +5')
    btn_rotate_cw = widgets.Button(description='Rot -5')
    btn_scale_up = widgets.Button(description='Bigger')
    btn_scale_down = widgets.Button(description='Smaller')

    smooth_flow_preview = widgets.Checkbox(value=True, description='Smooth mask/flow on predict', indent=False)
    smooth_flow_radius = widgets.FloatSlider(value=1.25, min=0.0, max=3.0, step=0.25, description='Flow smooth', continuous_update=False)
    line_width = widgets.IntSlider(value=6, min=2, max=18, step=1, description='Line', continuous_update=False)

    flow_field_selector = widgets.Dropdown(
        options=[(name, i) for i, name in enumerate(FLOW_FIELD_NAMES)],
        value=int(np.clip(flow_field_channel, 0, len(FLOW_FIELD_NAMES) - 1)),
        description='Flow field',
        disabled=not flow_enabled,
        layout=widgets.Layout(width='190px'),
    )

    drawing = {'active': False}

    wind_indicator.value = """
    <div style='font-family:sans-serif;width:92px;height:280px;display:flex;
                flex-direction:column;justify-content:center;align-items:center;
                color:#246;gap:8px'>
      <div style='font-size:12px;font-weight:700;color:#444'>Freestream</div>
      <svg width='78' height='164' viewBox='0 0 78 164' aria-label='wind direction'>
        <defs>
          <marker id='arrowhead' markerWidth='8' markerHeight='8' refX='7' refY='4'
                  orient='auto' markerUnits='strokeWidth'>
            <path d='M0,0 L8,4 L0,8 Z' fill='#1976d2'></path>
          </marker>
        </defs>
        <line x1='8' y1='24' x2='66' y2='24' stroke='#1976d2' stroke-width='4'
              stroke-linecap='round' marker-end='url(#arrowhead)'></line>
        <line x1='8' y1='58' x2='66' y2='58' stroke='#1976d2' stroke-width='4'
              stroke-linecap='round' marker-end='url(#arrowhead)' opacity='0.88'></line>
        <line x1='8' y1='92' x2='66' y2='92' stroke='#1976d2' stroke-width='4'
              stroke-linecap='round' marker-end='url(#arrowhead)' opacity='0.76'></line>
        <line x1='8' y1='126' x2='66' y2='126' stroke='#1976d2' stroke-width='4'
              stroke-linecap='round' marker-end='url(#arrowhead)' opacity='0.64'></line>
      </svg>
      <div style='font-size:12px;color:#555'>Mach 0.1</div>
    </div>"""

    def update_conditions(angle_deg=None):
        angle_text = 'not drawn yet' if angle_deg is None else f'{angle_deg:+.2f} deg'
        conditions.value = f"""
        <div style='font-family:sans-serif;margin:6px 0 8px 0'>
          <div style='font-size:14px;font-weight:700;margin-bottom:4px'>Starting Conditions</div>
          <table style='border-collapse:collapse;font-size:13px;color:#444'>
            <tr>
              <td style='padding:2px 14px 2px 0;font-weight:700'>Freestream Mach</td>
              <td style='padding:2px 0'>0.1</td>
            </tr>
            <tr>
              <td style='padding:2px 14px 2px 0;font-weight:700'>Reynolds number</td>
              <td style='padding:2px 0'>9,000,000</td>
            </tr>
            <tr>
              <td style='padding:2px 14px 2px 0;font-weight:700'>Estimated AoA</td>
              <td style='padding:2px 0'>{angle_text}</td>
            </tr>
          </table>
        </div>"""

    def apply_pen_style():
        canvas.stroke_style = 'white'
        canvas.line_width = int(line_width.value)
        canvas.line_cap = 'round'
        canvas.line_join = 'round'

    def reset_canvas():
        canvas.fill_style = 'black'
        canvas.fill_rect(0, 0, CANVAS_W, CANVAS_H)
        apply_pen_style()

    def get_canvas_rgba():
        return np.array(
            canvas.get_image_data(0, 0, CANVAS_W, CANVAS_H),
            dtype=np.uint8,
        )

    def set_canvas_rgba(rgba):
        canvas.put_image_data(rgba, 0, 0)

    def clear_prediction_output():
        with output:
            output.clear_output(wait=True)

    def refresh_angle_from_canvas():
        update_conditions(calculate_angle_of_attack(get_canvas_rgba()))

    def shift_canvas(dx, dy):
        rgba = get_canvas_rgba()
        shifted = np.zeros_like(rgba)
        shifted[:, :, 3] = 255

        src_x0 = max(0, -dx)
        src_x1 = min(CANVAS_W, CANVAS_W - dx)
        src_y0 = max(0, -dy)
        src_y1 = min(CANVAS_H, CANVAS_H - dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(CANVAS_W, CANVAS_W + dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(CANVAS_H, CANVAS_H + dy)

        if src_x0 < src_x1 and src_y0 < src_y1:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = rgba[src_y0:src_y1, src_x0:src_x1]

        set_canvas_rgba(shifted)
        refresh_angle_from_canvas()
        clear_prediction_output()

    def rotate_canvas(angle_deg):
        rgba = get_canvas_rgba()
        img = Image.fromarray(rgba, mode='RGBA')
        rotated = img.rotate(
            angle_deg,
            resample=Image.BICUBIC,
            center=(CANVAS_W / 2, CANVAS_H / 2),
            fillcolor=(0, 0, 0, 255),
        )
        set_canvas_rgba(np.array(rotated, dtype=np.uint8))
        refresh_angle_from_canvas()
        clear_prediction_output()

    def resize_airfoil(scale_factor):
        rgba = get_canvas_rgba()
        gray = rgba[:, :, 0]
        points_yx = np.argwhere(gray > 25)
        if len(points_yx) < 20:
            update_conditions()
            clear_prediction_output()
            return

        y0, x0 = points_yx.min(axis=0)
        y1, x1 = points_yx.max(axis=0) + 1
        crop = rgba[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]
        new_w = max(1, int(round(crop_w * scale_factor)))
        new_h = max(1, int(round(crop_h * scale_factor)))

        resized = Image.fromarray(crop, mode='RGBA').resize(
            (new_w, new_h),
            resample=Image.BICUBIC,
        )

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        paste_x0 = center_x - new_w // 2
        paste_y0 = center_y - new_h // 2
        paste_x1 = paste_x0 + new_w
        paste_y1 = paste_y0 + new_h

        dst_x0 = max(0, paste_x0)
        dst_y0 = max(0, paste_y0)
        dst_x1 = min(CANVAS_W, paste_x1)
        dst_y1 = min(CANVAS_H, paste_y1)
        src_x0 = dst_x0 - paste_x0
        src_y0 = dst_y0 - paste_y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        scaled = np.zeros_like(rgba)
        scaled[:, :, 3] = 255
        resized_rgba = np.array(resized, dtype=np.uint8)
        if dst_x0 < dst_x1 and dst_y0 < dst_y1:
            scaled[dst_y0:dst_y1, dst_x0:dst_x1] = resized_rgba[src_y0:src_y1, src_x0:src_x1]

        set_canvas_rgba(scaled)
        refresh_angle_from_canvas()
        clear_prediction_output()

    def on_mouse_down(x, y):
        drawing['active'] = True
        apply_pen_style()
        canvas.begin_path()
        canvas.move_to(x, y)

    def on_mouse_move(x, y):
        # Keep live drawing lightweight. Smoothing is only applied later to
        # the U-Net mask/flow preview when Predict is clicked.
        if not drawing['active']:
            return
        apply_pen_style()
        canvas.line_to(x, y)
        canvas.stroke()

    def on_mouse_up(x, y):
        if drawing['active']:
            canvas.line_to(x, y)
            canvas.stroke()
        drawing['active'] = False
        refresh_angle_from_canvas()

    def on_clear(_):
        reset_canvas()
        update_conditions()
        clear_prediction_output()

    def _make_coeff_html(mean_norm, std_norm, angle_deg):
        mean = mean_norm * scales_np
        std = std_norm * scales_np
        conf = 100.0 * np.abs(mean_norm) / (np.abs(mean_norm) + std_norm + 1e-9)

        rows = ''
        for name, mu, sigma, c in zip(LABELS, mean, std, conf):
            bar_w = int(c * 1.5)
            if c >= 70:
                bar_color = '#4caf50'
            elif c >= 40:
                bar_color = '#ff9800'
            else:
                bar_color = '#f44336'

            rows += f"""
            <tr style='border-bottom:1px solid #e0e0e0'>
              <td style='padding:6px 14px;font-weight:700;font-size:16px'>{name}</td>
              <td style='padding:6px 14px;text-align:right;font-size:15px'>{mu:+.4f}</td>
              <td style='padding:6px 14px;text-align:right;color:#888;font-size:13px'>&plusmn;{sigma:.4f}</td>
              <td style='padding:6px 14px'>
                <div style='display:flex;align-items:center;gap:8px'>
                  <div style='background:{bar_color};width:{bar_w}px;height:12px;border-radius:6px'></div>
                  <span style='font-size:13px;color:{bar_color};font-weight:600'>{c:.1f}%</span>
                </div>
              </td>
            </tr>"""

        return f"""
        <div style='font-family:sans-serif;margin-top:8px'>
          <div style='font-size:14px;font-weight:700;margin-bottom:4px'>
            Predicted Aerodynamic Coefficients
          </div>
          <div style='font-size:13px;color:#555;margin-bottom:6px'>
            Estimated angle of attack: {'not enough drawing data' if angle_deg is None else f'{angle_deg:+.2f} deg'}
          </div>
          <table style='border-collapse:collapse;width:100%'>
            <tr style='border-bottom:2px solid #bbb;color:#555;font-size:12px'>
              <th style='padding:4px 14px;text-align:left'>Coeff</th>
              <th style='padding:4px 14px;text-align:right'>Value</th>
              <th style='padding:4px 14px;text-align:right'>&plusmn;sigma</th>
              <th style='padding:4px 14px;text-align:left'>Confidence</th>
            </tr>
            {rows}
          </table>
          <div style='font-size:11px;color:#aaa;margin-top:6px'>
            MC Dropout - {n_mc} passes - conf = |mu| / (|mu| + sigma)
          </div>
        </div>"""

    def _make_flow_html(gray_model, angle_deg):
        if not flow_enabled:
            return ''

        field_channel = int(flow_field_selector.value)
        field_channel = max(0, min(field_channel, len(FLOW_FIELD_NAMES) - 1))
        field_name = FLOW_FIELD_NAMES[field_channel]

        flow = _predict_flow_field(
            flow_model=flow_model,
            gray_model=gray_model,
            angle_deg=angle_deg,
            flow_y_mean=flow_y_mean,
            flow_y_std=flow_y_std,
            field_channel=field_channel,
            mask_threshold=flow_mask_threshold,
            fill_holes=flow_fill_holes,
            clip_alpha=flow_clip_alpha,
            flip_y_for_model=flow_flip_y_for_model,
            flip_y_for_display=flow_flip_y_for_display,
        )

        field = flow['field']
        if np.isfinite(field).any():
            vmin = np.nanpercentile(field, 1)
            vmax = np.nanpercentile(field, 99)
        else:
            vmin, vmax = None, None

        flow_uri = _array_to_png_data_uri(
            field,
            cmap_name='viridis',
            vmin=vmin,
            vmax=vmax,
            title=f'Predicted {field_name}',
            origin='upper',
        )
        mask_uri = _array_to_png_data_uri(
            flow['solid_mask'],
            cmap_name='gray',
            vmin=0,
            vmax=1,
            title='U-Net solid mask',
            origin='upper',
        )

        

        clip_note = ''
        if angle_deg is not None and flow_clip_alpha and (angle_deg < 4.0 or angle_deg > 12.0):
            clip_note = f" Estimated AoA was clipped to {flow['alpha_used']:.2f}&deg; for the flow model."

        return f"""
        <div style='font-family:sans-serif;margin-top:14px;border-top:1px solid #ddd;padding-top:10px'>
          <div style='font-size:14px;font-weight:700;margin-bottom:4px'>
            Predicted Flow Field Preview
          </div>
          <div style='font-size:12px;color:#666;margin-bottom:8px'>
            Showing <b>{field_name}</b>. U-Net input uses drawn solid mask + normalized AoA.
            {'Flow mask is flipped vertically for the model, then flipped back for display.' if flow_flip_y_for_model and flow_flip_y_for_display else ''}
            {clip_note}
          </div>
          <div style='display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap'>
            <div><img src='{mask_uri}' style='max-width:260px;border:1px solid #ddd;border-radius:6px'></div>
            <div><img src='{flow_uri}' style='max-width:300px;border:1px solid #ddd;border-radius:6px'></div>
          </div>
        </div>"""

    def on_predict(_):
        rgba = get_canvas_rgba()
        gray_raw = rgba[:, :, 0]

        # AoA and coefficient prediction use the raw drawing.
        angle_deg = calculate_angle_of_attack(gray_raw)
        gray_model_coeff = _resize_gray_to_model(gray_raw)

        x_coeff = gray_model_coeff[np.newaxis, :, :, np.newaxis].astype('float32')
        mean_norm, std_norm = _mc_predict(model, x_coeff, n_mc)

        # U-Net mask/flow preview may use a smoothed drawing.
        # This does not change the displayed canvas and does not affect coeffs.
        if smooth_flow_preview.value:
            gray_for_flow = _smooth_gray_for_flow_preview(
                gray_raw,
                radius=smooth_flow_radius.value,
            )
        else:
            gray_for_flow = gray_raw

        gray_model_flow = _resize_gray_to_model(gray_for_flow)

        html = _make_coeff_html(mean_norm, std_norm, angle_deg)
        html += _make_flow_html(gray_model_flow, angle_deg)

        with output:
            output.clear_output(wait=True)
            display(HTML(html))

    def on_line_width_change(_):
        apply_pen_style()

    def on_flow_selector_change(_):
        clear_prediction_output()

    reset_canvas()
    update_conditions()

    # Clear any pre-existing callbacks to prevent duplicates when the cell is re-run.
    for _btn in (btn_predict, btn_clear, btn_up, btn_down, btn_left, btn_right,
                 btn_rotate_ccw, btn_rotate_cw, btn_scale_up, btn_scale_down):
        _btn._click_handlers.callbacks.clear()
    line_width.unobserve_all()
    flow_field_selector.unobserve_all()
    canvas._mouse_down_callbacks.callbacks.clear()
    canvas._mouse_move_callbacks.callbacks.clear()
    canvas._mouse_up_callbacks.callbacks.clear()

    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)
    btn_clear.on_click(on_clear)
    btn_predict.on_click(on_predict)
    btn_up.on_click(lambda _: shift_canvas(0, -10))
    btn_down.on_click(lambda _: shift_canvas(0, 10))
    btn_left.on_click(lambda _: shift_canvas(-10, 0))
    btn_right.on_click(lambda _: shift_canvas(10, 0))
    btn_rotate_ccw.on_click(lambda _: rotate_canvas(5))
    btn_rotate_cw.on_click(lambda _: rotate_canvas(-5))
    btn_scale_up.on_click(lambda _: resize_airfoil(1.1))
    btn_scale_down.on_click(lambda _: resize_airfoil(0.9))
    line_width.observe(on_line_width_change, names='value')
    flow_field_selector.observe(on_flow_selector_change, names='value')

    transform_controls = widgets.GridBox(
        [
            btn_left,
            btn_right,
            btn_up,
            btn_down,
            btn_rotate_cw,
            btn_rotate_ccw,
            btn_scale_down,
            btn_scale_up,
        ],
        layout=widgets.Layout(
            grid_template_columns='repeat(4, 90px)',
            grid_template_rows='repeat(2, 32px)',
            grid_gap='6px',
        ),
    )

    draw_controls = widgets.HBox([line_width, smooth_flow_preview, smooth_flow_radius])
    flow_controls = widgets.HBox([flow_field_selector]) if flow_enabled else widgets.HTML(
        "<div style='font-family:sans-serif;font-size:12px;color:#888'>Flow model not loaded.</div>"
    )

    clear_output(wait=True)
    display(conditions)
    display(widgets.HBox([btn_clear, btn_predict]))
    display(widgets.HTML(
        "<div style='font-family:sans-serif;font-size:13px;font-weight:700;"
        "margin:8px 0 4px 0;color:#444'>"
        "Rotate and Move your Wing Design Here"
        "</div>"
    ))
    display(transform_controls)
    display(widgets.HTML(
        "<div style='font-family:sans-serif;font-size:13px;font-weight:700;"
        "margin:8px 0 4px 0;color:#444'>"
        "Drawing / Flow Smoothing Settings"
        "</div>"
    ))
    display(draw_controls)
    display(widgets.HTML(
        "<div style='font-family:sans-serif;font-size:13px;font-weight:700;"
        "margin:8px 0 4px 0;color:#444'>"
        "Flow Preview Settings"
        "</div>"
    ))
    display(flow_controls)
    display(widgets.HBox([wind_indicator, canvas], layout=widgets.Layout(align_items='center')))
    display(output)

    return {
        'canvas': canvas,
        'conditions': conditions,
        'wind_indicator': wind_indicator,
        'output': output,
        'btn_predict': btn_predict,
        'btn_clear': btn_clear,
        'btn_up': btn_up,
        'btn_down': btn_down,
        'btn_left': btn_left,
        'btn_right': btn_right,
        'btn_rotate_ccw': btn_rotate_ccw,
        'btn_rotate_cw': btn_rotate_cw,
        'btn_scale_up': btn_scale_up,
        'btn_scale_down': btn_scale_down,
        'smooth_flow_preview': smooth_flow_preview,
        'smooth_flow_radius': smooth_flow_radius,
        'line_width': line_width,
        'flow_field_selector': flow_field_selector,
        'flow_flip_y_for_model': flow_flip_y_for_model,
        'flow_flip_y_for_display': flow_flip_y_for_display,
        'calculate_angle_of_attack': calculate_angle_of_attack,
    }
