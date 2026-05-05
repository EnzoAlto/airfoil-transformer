import numpy as np
from PIL import Image
from IPython.display import HTML, clear_output, display
import ipywidgets as widgets
from ipycanvas import Canvas


CANVAS_W, CANVAS_H = 280, 280
MODEL_W, MODEL_H = 128, 128
LABELS = ['Cl', 'Cd', 'Cm']


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


def create_airfoil_draw_widget(model, scales_np, n_mc=50):
    """
    Create and display a drawable airfoil prediction widget.

    Parameters
    ----------
    model : keras model
        Model that accepts input shape (1, 128, 128, 1).
    scales_np : array-like
        Output scaling values for [Cl, Cd, Cm].
    n_mc : int
        Number of MC dropout prediction passes.
    """
    scales_np = np.asarray(scales_np, dtype=np.float32)

    canvas = Canvas(width=CANVAS_W, height=CANVAS_H, sync_image_data=True)
    conditions = widgets.HTML()
    wind_indicator = widgets.HTML()
    output = widgets.Output()
    btn_predict = widgets.Button(description='Predict', button_style='primary')
    btn_clear = widgets.Button(description='Clear')
    btn_up = widgets.Button(description='Up')
    btn_down = widgets.Button(description='Down')
    btn_left = widgets.Button(description='Left')
    btn_right = widgets.Button(description='Right')
    btn_rotate_ccw = widgets.Button(description='Rot +5')
    btn_rotate_cw = widgets.Button(description='Rot -5')
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

    def reset_canvas():
        canvas.fill_style = 'black'
        canvas.fill_rect(0, 0, CANVAS_W, CANVAS_H)
        canvas.stroke_style = 'white'
        canvas.line_width = 6
        canvas.line_cap = 'round'
        canvas.line_join = 'round'

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

    def on_mouse_down(x, y):
        drawing['active'] = True
        canvas.begin_path()
        canvas.move_to(x, y)

    def on_mouse_move(x, y):
        if not drawing['active']:
            return
        canvas.line_to(x, y)
        canvas.stroke()

    def on_mouse_up(x, y):
        drawing['active'] = False
        refresh_angle_from_canvas()

    def on_clear(_):
        reset_canvas()
        update_conditions()
        clear_prediction_output()

    def on_predict(_):
        rgba = get_canvas_rgba()
        gray = rgba[:, :, 0]
        angle_deg = calculate_angle_of_attack(gray)
        img = Image.fromarray(gray, mode='L').resize((MODEL_W, MODEL_H), Image.LANCZOS)
        x = np.array(img).astype('float32') / 255.0
        x = x[np.newaxis, :, :, np.newaxis]

        mean_norm, std_norm = _mc_predict(model, x, n_mc)
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

        html = f"""
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

        with output:
            output.clear_output(wait=True)
            display(HTML(html))

    reset_canvas()
    update_conditions()
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

    transform_controls = widgets.HBox([
        btn_left,
        btn_right,
        btn_up,
        btn_down,
        btn_rotate_cw,
        btn_rotate_ccw,
    ])

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
        'calculate_angle_of_attack': calculate_angle_of_attack,
    }
