"""
Gradio Airfoil Surrogate App

Run:
    pip install gradio tensorflow pillow matplotlib scipy
    python app.py

Expected model files:
    models/airfoil_vgg16_best1_1k.keras
    models/airfoil_unet_all_fields.keras
"""

import os
import io

import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import binary_fill_holes
except Exception:
    binary_fill_holes = None


CANVAS_W, CANVAS_H = 280, 280
MODEL_W, MODEL_H = 128, 128

LABELS = ["Cl", "Cd", "Cm"]
FLOW_FIELD_NAMES = ("rho", "rho_u", "rho_v", "e")
COEFF_DISPLAY_NAMES = {
    "Cl": "Coefficient of Lift, Cl",
    "Cd": "Coefficient of Drag, Cd",
    "Cm": "Coefficient of Moment, Cm",
}
COEFF_DECIMALS = {"Cl": 2, "Cd": 4, "Cm": 3}

CL_SCALE, CD_SCALE, CM_SCALE = (2.0999999999999996, 0.033, 0.34)

eps = 1e-8
scales_np = np.array([CL_SCALE, CD_SCALE, CM_SCALE], dtype=np.float32)
scales_np = np.where(scales_np < eps, 1.0, scales_np)

Y_mean = np.array(
    [[[[0.9994531, 0.09668528, 0.0247279, 1.789871]]]],
    dtype=np.float32,
)
Y_std = np.array(
    [[[[0.00245761, 0.02148222, 0.02246827, 0.00371735]]]],
    dtype=np.float32,
)

COEFF_MODEL_PATH = "models/airfoil_vgg16_best1_1k.keras"
FLOW_MODEL_PATH = "models/airfoil_unet_all_fields.keras"

# Fixed app settings
DEFAULT_N_MC = 50
DEFAULT_FLOW_MASK_THRESHOLD = 0.10
DEFAULT_FILL_HOLES = True
DEFAULT_FLIP_Y_FOR_MODEL = True
DEFAULT_FLIP_Y_FOR_DISPLAY = True
DEFAULT_CLIP_ALPHA = True
DEFAULT_ALPHA_MIN = 4.0
DEFAULT_ALPHA_MAX = 12.0
FREESTREAM_MACH = 0.1
REYNOLDS_NUMBER = 9_000_000


def normalized_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred) / scales_np)


def load_models():
    if not os.path.exists(COEFF_MODEL_PATH):
        raise FileNotFoundError(f"Coefficient model not found: {COEFF_MODEL_PATH}")
    if not os.path.exists(FLOW_MODEL_PATH):
        raise FileNotFoundError(f"Flow model not found: {FLOW_MODEL_PATH}")

    coeff_model = keras.models.load_model(
        COEFF_MODEL_PATH,
        custom_objects={"normalized_mae": normalized_mae},
        compile=False,
    )

    flow_model = keras.models.load_model(
        FLOW_MODEL_PATH,
        compile=False,
    )

    return coeff_model, flow_model


COEFF_MODEL, FLOW_MODEL = load_models()


def make_blank_canvas():
    return Image.new("RGB", (CANVAS_W, CANVAS_H), color=(0, 0, 0))


def make_editor_image(img):
    """
    Return a PIL image suitable for Gradio ImageEditor.

    Keeping this as a plain image is the most version-tolerant option for
    ImageEditor when type="pil".
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert("RGBA")


def transform_editor_image(editor_value, mode):
    """
    Apply button transforms to the drawn airfoil image.

    Supported modes:
        clear, left, right, up, down, rot_ccw, rot_cw, bigger, smaller
    """
    if mode == "clear":
        return make_editor_image(make_blank_canvas())

    img = editor_value_to_pil(editor_value).convert("RGBA")
    arr = np.asarray(img).copy()

    if mode in {"left", "right", "up", "down"}:
        if mode == "left":
            dx, dy = -10, 0
        elif mode == "right":
            dx, dy = 10, 0
        elif mode == "up":
            dx, dy = 0, -10
        else:
            dx, dy = 0, 10

        shifted = np.zeros_like(arr)
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
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

        return make_editor_image(Image.fromarray(shifted, mode="RGBA"))

    if mode in {"rot_ccw", "rot_cw"}:
        angle = 5 if mode == "rot_ccw" else -5
        rotated = img.rotate(
            angle,
            resample=Image.BICUBIC,
            center=(CANVAS_W / 2, CANVAS_H / 2),
            fillcolor=(0, 0, 0, 255),
        )
        return make_editor_image(rotated)

    if mode in {"bigger", "smaller"}:
        scale_factor = 1.1 if mode == "bigger" else 0.9

        gray = pil_to_gray_airfoil(img)
        points_yx = np.argwhere(gray > 25)

        if len(points_yx) < 20:
            return make_editor_image(img)

        y0, x0 = points_yx.min(axis=0)
        y1, x1 = points_yx.max(axis=0) + 1

        crop = img.crop((x0, y0, x1, y1))
        crop_w, crop_h = crop.size

        new_w = max(1, int(round(crop_w * scale_factor)))
        new_h = max(1, int(round(crop_h * scale_factor)))

        resized = crop.resize((new_w, new_h), resample=Image.BICUBIC)

        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2

        paste_x0 = center_x - new_w // 2
        paste_y0 = center_y - new_h // 2
        paste_x1 = paste_x0 + new_w
        paste_y1 = paste_y0 + new_h

        out = Image.new("RGBA", (CANVAS_W, CANVAS_H), color=(0, 0, 0, 255))

        dst_x0 = max(0, paste_x0)
        dst_y0 = max(0, paste_y0)
        dst_x1 = min(CANVAS_W, paste_x1)
        dst_y1 = min(CANVAS_H, paste_y1)

        src_x0 = dst_x0 - paste_x0
        src_y0 = dst_y0 - paste_y0
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x0 < dst_x1 and dst_y0 < dst_y1:
            resized_crop = resized.crop((src_x0, src_y0, src_x1, src_y1))
            out.paste(resized_crop, (dst_x0, dst_y0), resized_crop)

        return make_editor_image(out)

    return make_editor_image(img)


def editor_value_to_pil(editor_value):
    """
    Gradio ImageEditor/Sketchpad generally passes a dict with:
    background, layers, composite.
    """
    if editor_value is None:
        return make_blank_canvas()

    if isinstance(editor_value, dict):
        img = editor_value.get("composite", None)

        if img is None:
            layers = editor_value.get("layers", [])
            if layers:
                img = layers[-1]
            else:
                img = editor_value.get("background", None)

        if img is None:
            return make_blank_canvas()
    else:
        img = editor_value

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(img, str):
        img = Image.open(img)

    if not isinstance(img, Image.Image):
        return make_blank_canvas()

    return img.convert("RGBA")


def pil_to_gray_airfoil(img):
    img = img.convert("RGBA")
    arr = np.asarray(img).astype(np.uint8)

    rgb = arr[..., :3]
    alpha = arr[..., 3:4].astype(np.float32) / 255.0

    rgb_on_black = (rgb.astype(np.float32) * alpha).astype(np.uint8)
    gray = np.max(rgb_on_black, axis=-1).astype(np.uint8)

    return gray


def calculate_angle_of_attack(image_data, threshold=25, min_pixels=20):
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


def resize_gray_to_model(gray):
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    img = img.resize((MODEL_W, MODEL_H), Image.LANCZOS)
    return np.asarray(img).astype("float32") / 255.0


def smooth_gray_for_flow(gray, radius=1.25):
    radius = float(radius)
    if radius <= 0:
        return np.asarray(gray, dtype=np.uint8)

    img = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.uint8)


def make_solid_mask_from_gray(gray_model, threshold=0.10, fill_holes=True):
    mask = gray_model > float(threshold)

    if fill_holes and binary_fill_holes is not None:
        mask = binary_fill_holes(mask)

    return mask.astype("float32")


def flow_alpha_channel(angle_deg, solid_mask):
    if angle_deg is None:
        raw_alpha = DEFAULT_ALPHA_MIN
    else:
        # Use AoA magnitude for the flow model.
        # Example: -6 deg -> +6 deg, then clip to the trained 4-12 deg range.
        raw_alpha = abs(float(angle_deg))

    alpha_used = float(np.clip(raw_alpha, DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX))
    alpha_norm = (alpha_used - 8.0) / 4.0
    alpha_channel = np.ones_like(solid_mask, dtype=np.float32) * alpha_norm

    return alpha_channel, raw_alpha, alpha_used, alpha_norm


def mc_predict(model, x, n_mc=50):
    runs = np.stack(
        [model(x, training=True).numpy() for _ in range(int(n_mc))],
        axis=0,
    )
    return runs.mean(axis=0)[0], runs.std(axis=0)[0]


def predict_flow_field(
    flow_model,
    gray_model,
    angle_deg,
    field_channel=0,
    mask_threshold=DEFAULT_FLOW_MASK_THRESHOLD,
    fill_holes=DEFAULT_FILL_HOLES,
    flip_y_for_model=DEFAULT_FLIP_Y_FOR_MODEL,
    flip_y_for_display=DEFAULT_FLIP_Y_FOR_DISPLAY,
):
    solid_mask_display = make_solid_mask_from_gray(
        gray_model,
        threshold=mask_threshold,
        fill_holes=fill_holes,
    )

    solid_mask_model = np.flipud(solid_mask_display) if flip_y_for_model else solid_mask_display

    alpha_channel, raw_alpha, alpha_used, alpha_norm = flow_alpha_channel(
        angle_deg,
        solid_mask_model,
    )

    x_flow = np.stack([solid_mask_model, alpha_channel], axis=-1)[np.newaxis].astype("float32")

    y_pred_norm = flow_model.predict(x_flow, verbose=0)
    y_pred_model = y_pred_norm * Y_std + Y_mean

    field_model = y_pred_model[0, :, :, field_channel].astype(np.float32)
    field_model[solid_mask_model > 0.5] = np.nan

    if flip_y_for_display:
        field_display = np.flipud(field_model)
        solid_mask_preview = solid_mask_display
    else:
        field_display = field_model
        solid_mask_preview = solid_mask_model

    field_display[solid_mask_preview > 0.5] = np.nan

    return {
        "field": field_display,
        "solid_mask": solid_mask_preview,
        "raw_alpha": raw_alpha,
        "alpha_used": alpha_used,
        "alpha_norm": alpha_norm,
    }


def array_to_figure_pil(
    arr,
    cmap_name="viridis",
    vmin=None,
    vmax=None,
    title=None,
    origin="upper",
    add_colorbar=True,
):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    masked = np.ma.masked_invalid(arr)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    im = ax.imshow(
        masked,
        cmap=cmap,
        origin=origin,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=10)

    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf).convert("RGB")


def make_coeff_dataframe(mean_norm, std_norm):
    mean = mean_norm * scales_np
    conf = 100.0 * np.abs(mean_norm) / (np.abs(mean_norm) + std_norm + 1e-9)

    rows = []
    for name, mu, c in zip(LABELS, mean, conf):
        rows.append([
            COEFF_DISPLAY_NAMES[name],
            round(float(mu), COEFF_DECIMALS[name]),
            round(float(c), 0),
        ])

    return rows


def predict_airfoil(
    sketch,
    flow_field_name,
    smooth_mask_flow,
    flow_smooth_radius,
):
    img = editor_value_to_pil(sketch)
    gray_raw = pil_to_gray_airfoil(img)

    angle_deg = calculate_angle_of_attack(gray_raw)

    gray_model_coeff = resize_gray_to_model(gray_raw)
    x_coeff = gray_model_coeff[np.newaxis, :, :, np.newaxis].astype("float32")

    mean_norm, std_norm = mc_predict(COEFF_MODEL, x_coeff, n_mc=DEFAULT_N_MC)
    coeff_rows = make_coeff_dataframe(mean_norm, std_norm)

    if smooth_mask_flow:
        gray_for_flow = smooth_gray_for_flow(gray_raw, radius=flow_smooth_radius)
    else:
        gray_for_flow = gray_raw

    gray_model_flow = resize_gray_to_model(gray_for_flow)
    field_channel = FLOW_FIELD_NAMES.index(flow_field_name)

    flow = predict_flow_field(
        flow_model=FLOW_MODEL,
        gray_model=gray_model_flow,
        angle_deg=angle_deg,
        field_channel=field_channel,
        mask_threshold=DEFAULT_FLOW_MASK_THRESHOLD,
        fill_holes=DEFAULT_FILL_HOLES,
        flip_y_for_model=DEFAULT_FLIP_Y_FOR_MODEL,
        flip_y_for_display=DEFAULT_FLIP_Y_FOR_DISPLAY,
    )

    field = flow["field"]
    if np.isfinite(field).any():
        vmin = np.nanpercentile(field, 1)
        vmax = np.nanpercentile(field, 99)
    else:
        vmin, vmax = None, None

    mask_img = array_to_figure_pil(
        flow["solid_mask"],
        cmap_name="gray",
        vmin=0,
        vmax=1,
        title="U-Net solid mask",
        origin="upper",
        add_colorbar=False,
    )

    flow_img = array_to_figure_pil(
        field,
        cmap_name="viridis",
        vmin=vmin,
        vmax=vmax,
        title=f"Predicted {flow_field_name}",
        origin="upper",
        add_colorbar=True,
    )

    if angle_deg is None:
        aoa_text = "Calculated AoA from Drawing: not enough drawing data"
    else:
        aoa_text = f"Calculated AoA from Drawing: {angle_deg:+.2f}°"

        return coeff_rows, aoa_text, mask_img, flow_img


def build_app():
    with gr.Blocks(
        title="Airfoil Coefficient + Flow Field Surrogate",
        css="""
        #airfoil-canvas-row {
            gap: 15px;
        }
        #wind-panel {
            flex: 0 0 105px;
            min-width: 105px;
        }
        """,
    ) as demo:
        gr.Markdown(
            """
            # 2D Airfoil Simulation with Neural Networks

            Draw a **white airfoil on the black canvas**, then click **Predict**.

            Outputs:
            - aerodynamic coefficients: `Cl`, `Cd`, `Cm`
            - U-Net solid mask
            - selected flow-field preview: `rho`, `rho_u`, `rho_v`, or `e`
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    f"""
                    <div style="font-family:sans-serif;margin-bottom:10px">
                      <div style="font-size:18px;font-weight:700;margin-bottom:8px">Simulation Conditions</div>
                      <div><b>Reynolds Number:</b> {REYNOLDS_NUMBER:,}</div>
                      <div><b>Freestream Mach:</b> {FREESTREAM_MACH}</div>
                      <div><b>Freestream Direction:</b> Left → Right</div>
                    </div>
                    """
                )

                with gr.Row(elem_id="airfoil-canvas-row"):
                    gr.HTML(
                        """
                        <div style="width:105px;height:360px;display:flex;flex-direction:column;
                                    justify-content:center;align-items:center;font-family:sans-serif;
                                    border:1px solid #ddd;border-radius:10px;background:#fafafa">
                          <div style="font-size:15px;font-weight:700;color:#444;margin-bottom:10px">
                            Wind
                          </div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          <div style="font-size:40px;color:#1976d2;line-height:1.25">➜</div>
                          </div>
                        </div>
                        """,
                        elem_id="wind-panel",
                    )

                    sketch = gr.ImageEditor(
                        value=make_blank_canvas(),
                        image_mode="RGBA",
                        type="pil",
                        label="Draw airfoil",
                        height=360,
                        width=360,
                        canvas_size=(CANVAS_W, CANVAS_H),
                        fixed_canvas=True,
                        sources=(),
                        transforms=(),
                        layers=False,
                        brush=gr.Brush(
                            default_size=6,
                            colors=["#ffffff"],
                            default_color="#ffffff",
                            color_mode="fixed",
                        ),
                        eraser=gr.Eraser(default_size=14),
                    )

                predict_btn = gr.Button("Predict", variant="primary")
                clear_drawing_btn = gr.Button("Clear Drawing")


                gr.Markdown("### Transform Drawing")
                with gr.Row():
                    left_btn = gr.Button("Left")
                    right_btn = gr.Button("Right")
                    up_btn = gr.Button("Up")
                    down_btn = gr.Button("Down")

                with gr.Row():
                    rot_cw_btn = gr.Button("Rot -5")
                    rot_ccw_btn = gr.Button("Rot +5")
                    smaller_btn = gr.Button("Smaller")
                    bigger_btn = gr.Button("Bigger")

                gr.Markdown("### Flow Settings")
                flow_field = gr.Dropdown(
                    choices=list(FLOW_FIELD_NAMES),
                    value="rho",
                    label="Flow field to display",
                )

                with gr.Accordion("Advanced settings", open=False):
                    smooth_mask_flow = gr.Checkbox(
                        value=True,
                        label="Smooth mask/flow on predict",
                    )
                    flow_smooth_radius = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=1.25,
                        step=0.25,
                        label="Flow smooth radius",
                    )

            with gr.Column(scale=1):
                aoa_text = gr.Textbox(
                    label="Detected / applied AoA",
                    interactive=False,
                )
                coeff_table = gr.Dataframe(
                    headers=["Coefficient", "Mean", "Confidence %"],
                    datatype=["str", "number", "number"],
                    label="Predicted aerodynamic coefficients",
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        mask_output = gr.Image(
                            label="Solid mask used by U-Net",
                            type="pil",
                        )

                    with gr.Column(scale=2):
                        flow_output = gr.Image(
                            label="Predicted flow field",
                            type="pil",
                        )

        gr.HTML(
            """
            <section style="
                margin-top: 26px;
                padding: 8px 0 0;
                font-family: sans-serif;
            ">
              <div style="font-size:18px;font-weight:700;margin-bottom:12px;color:#ffffff">
                Dataset Output Guide
              </div>
              <div style="
                  display:grid;
                  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                  gap: 16px;
              ">
                <div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px">
                  <div style="font-size:15px;font-weight:700;margin-bottom:8px;color:#ffffff">
                    Aerodynamic Coefficients
                  </div>
                  <ul style="margin:0;padding-left:18px;line-height:1.55;color:#374151">
                    <li><b>C<sub>d</sub></b>: representing the coefficient of drag for each airfoil geometry at the specified angle of attack.</li>
                    <li><b>C<sub>l</sub></b>: representing the coefficient of lift for each airfoil geometry at the specified angle of attack.</li>
                    <li><b>C<sub>m</sub></b>: representing the coefficient of moment for each airfoil geometry at the specified angle of attack.</li>
                  </ul>
                </div>
                <div style="border:1px solid #e5e7eb;border-radius:8px;padding:14px">
                  <div style="font-size:15px;font-weight:700;margin-bottom:8px;color:#ffffff">
                    Flow Field Variables
                  </div>
                  <ul style="margin:0;padding-left:18px;line-height:1.55;color:#374151">
                    <li><b>rho</b>: array representing the flow density at each mesh point.</li>
                    <li><b>rho_u</b>: array representing the flow momentum in the x-direction at each mesh point.</li>
                    <li><b>rho_v</b>: array representing the flow momentum in the y-direction at each mesh point.</li>
                    <li><b>e</b>: array representing the total energy in the flow at each mesh point.</li>
                  </ul>
                </div>
              </div>
            </section>
            """
        )

        clear_drawing_btn.click(
            fn=lambda current: transform_editor_image(current, "clear"),
            inputs=[sketch],
            outputs=[sketch],
        )
        left_btn.click(
            fn=lambda current: transform_editor_image(current, "left"),
            inputs=[sketch],
            outputs=[sketch],
        )
        right_btn.click(
            fn=lambda current: transform_editor_image(current, "right"),
            inputs=[sketch],
            outputs=[sketch],
        )
        up_btn.click(
            fn=lambda current: transform_editor_image(current, "up"),
            inputs=[sketch],
            outputs=[sketch],
        )
        down_btn.click(
            fn=lambda current: transform_editor_image(current, "down"),
            inputs=[sketch],
            outputs=[sketch],
        )
        rot_cw_btn.click(
            fn=lambda current: transform_editor_image(current, "rot_cw"),
            inputs=[sketch],
            outputs=[sketch],
        )
        rot_ccw_btn.click(
            fn=lambda current: transform_editor_image(current, "rot_ccw"),
            inputs=[sketch],
            outputs=[sketch],
        )
        smaller_btn.click(
            fn=lambda current: transform_editor_image(current, "smaller"),
            inputs=[sketch],
            outputs=[sketch],
        )
        bigger_btn.click(
            fn=lambda current: transform_editor_image(current, "bigger"),
            inputs=[sketch],
            outputs=[sketch],
        )

        predict_btn.click(
            fn=predict_airfoil,
            inputs=[
                sketch,
                flow_field,
                smooth_mask_flow,
                flow_smooth_radius,
            ],
            outputs=[
                coeff_table,
                aoa_text,
                mask_output,
                flow_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
