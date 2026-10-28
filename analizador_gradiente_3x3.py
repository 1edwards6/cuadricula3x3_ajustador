import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


st.set_page_config(page_title="Analizador de cuadrícula por gradiente", layout="wide")
st.title("Analizador de imágenes por gradiente — cuadrícula 3×3")
st.write(
    "Busca la región central donde la intensidad cambia más rápidamente entre píxeles. "
    "Puedes ubicar la cuadrícula automáticamente y después ajustar sus líneas."
)


# --- Image and gradient utilities -------------------------------------------------
def to_gray_array(image: Image.Image, blur_radius: float) -> np.ndarray:
    """Convert image to grayscale and optionally smooth it to reduce noise."""
    gray = ImageOps.grayscale(image.convert("RGB"))
    if blur_radius > 0:
        gray = gray.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return np.asarray(gray, dtype=np.float32)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute the gradient magnitude sqrt((dI/dx)^2 + (dI/dy)^2)."""
    grad_y, grad_x = np.gradient(gray)
    return np.hypot(grad_x, grad_y)


def integral_image(values: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero-padded first row and column."""
    return np.pad(values, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def rectangle_mean(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    total = (
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(total / area)


def cell_means(integral: np.ndarray, width: int, height: int,
               x1_pct: int, x2_pct: int, y1_pct: int, y2_pct: int):
    """Return means for the nine grid cells, measured in gradient magnitude."""
    xs = [0, round(width * x1_pct / 100), round(width * x2_pct / 100), width]
    ys = [0, round(height * y1_pct / 100), round(height * y2_pct / 100), height]
    xs = [max(0, min(width, int(v))) for v in xs]
    ys = [max(0, min(height, int(v))) for v in ys]
    values = []
    for row in range(3):
        current_row = []
        for col in range(3):
            current_row.append(
                rectangle_mean(integral, xs[col], ys[row], xs[col + 1], ys[row + 1])
            )
        values.append(current_row)
    return values, xs, ys


def find_best_grid(gradient: np.ndarray):
    """Search grid positions and favor a high-gradient central cell.

    Constraints: outer bands >= 10% and the central cell >= 20% of image
    width/height. Positions are explored in 5% increments.
    """
    height, width = gradient.shape
    integral = integral_image(gradient)
    pairs = []
    for first in range(10, 80, 5):
        for second in range(first + 20, 91, 5):
            if second <= 90:
                pairs.append((first, second))

    best_score = -float("inf")
    best_positions = (35, 65, 35, 65)
    best_center = 0.0
    best_outer = 0.0

    for x1_pct, x2_pct in pairs:
        for y1_pct, y2_pct in pairs:
            means, _, _ = cell_means(
                integral, width, height, x1_pct, x2_pct, y1_pct, y2_pct
            )
            center = means[1][1]
            outer = [means[r][c] for r in range(3) for c in range(3) if (r, c) != (1, 1)]
            outer_mean = float(np.mean(outer))
            # A heuristic: seek high local change in the central cell, while
            # mildly rewarding contrast against the surrounding cells.
            score = center - 0.35 * outer_mean
            if score > best_score:
                best_score = score
                best_positions = (x1_pct, x2_pct, y1_pct, y2_pct)
                best_center = center
                best_outer = outer_mean

    return best_positions, best_score, best_center, best_outer


def draw_grid(image: Image.Image, positions):
    x1_pct, x2_pct, y1_pct, y2_pct = positions
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    x1, x2 = round(width * x1_pct / 100), round(width * x2_pct / 100)
    y1, y2 = round(height * y1_pct / 100), round(height * y2_pct / 100)
    line_width = max(2, round(min(width, height) / 300))
    # Draw outer lines first, then emphasize the central cell in red.
    for x in (x1, x2):
        draw.line((x, 0, x, height), fill=(30, 220, 255), width=line_width)
    for y in (y1, y2):
        draw.line((0, y, width, y), fill=(30, 220, 255), width=line_width)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 65, 65), width=line_width + 1)
    return result


def make_gradient_preview(gradient: np.ndarray) -> Image.Image:
    """Create a high-contrast grayscale visualization of the gradient map."""
    high = float(np.percentile(gradient, 99))
    if high <= 1e-9:
        scaled = np.zeros_like(gradient, dtype=np.uint8)
    else:
        scaled = np.clip(gradient / high * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


# --- Session state ----------------------------------------------------------------
defaults = {"x1_pct": 35, "x2_pct": 65, "y1_pct": 35, "y2_pct": 65}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

uploaded = st.file_uploader(
    "Carga una imagen", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("Carga una imagen para analizar el gradiente y ubicar la cuadrícula.")
    st.stop()

try:
    original = Image.open(uploaded).convert("RGB")
    original.load()
except Exception as exc:
    st.error(f"No fue posible abrir la imagen: {exc}")
    st.stop()

image_signature = (uploaded.name, uploaded.size)
if st.session_state.get("image_signature") != image_signature:
    st.session_state["image_signature"] = image_signature
    for key, value in defaults.items():
        st.session_state[key] = value

with st.sidebar:
    st.header("Configuración")
    blur_radius = st.slider(
        "Suavizado previo (reduce sensibilidad al ruido)",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.2,
        help="Un suavizado pequeño reduce la influencia del ruido sin eliminar necesariamente los bordes principales.",
    )
    st.caption("La magnitud del gradiente mide cuánto cambia la intensidad de un píxel respecto a sus vecinos.")

# Resizing is only for faster analysis; display keeps the original resolution.
analysis_image = original.copy()
max_side = 1200
if max(analysis_image.size) > max_side:
    scale = max_side / max(analysis_image.size)
    analysis_image = analysis_image.resize(
        (max(2, round(analysis_image.width * scale)), max(2, round(analysis_image.height * scale))),
        Image.Resampling.LANCZOS,
    )

gray = to_gray_array(analysis_image, blur_radius)
gradient = gradient_magnitude(gray)
gradient_integral = integral_image(gradient)

button_col, info_col = st.columns([1, 2])
with button_col:
    if st.button("Encontrar automáticamente la zona de mayor gradiente", type="primary", use_container_width=True):
        positions, score, center_mean, outer_mean = find_best_grid(gradient)
        st.session_state["x1_pct"], st.session_state["x2_pct"], st.session_state["y1_pct"], st.session_state["y2_pct"] = positions
        st.session_state["last_auto"] = {
            "score": score,
            "center": center_mean,
            "outer": outer_mean,
        }
        st.rerun()
with info_col:
    st.caption(
        "La búsqueda automática explora posiciones cada 5%. Mantiene las bandas exteriores con al menos 10% "
        "del ancho/alto y la celda central con al menos 20%."
    )

# Keep the right boundary valid when the left boundary is moved.
if st.session_state["x2_pct"] < st.session_state["x1_pct"] + 20:
    st.session_state["x2_pct"] = min(90, st.session_state["x1_pct"] + 20)
if st.session_state["y2_pct"] < st.session_state["y1_pct"] + 20:
    st.session_state["y2_pct"] = min(90, st.session_state["y1_pct"] + 20)

st.subheader("Ajuste manual de la cuadrícula")
left, right = st.columns(2)
with left:
    st.slider("Línea vertical izquierda (%)", 10, 70, step=5, key="x1_pct")
    # Update the valid range before rendering the dependent slider.
    if st.session_state["x2_pct"] < st.session_state["x1_pct"] + 20:
        st.session_state["x2_pct"] = min(90, st.session_state["x1_pct"] + 20)
    st.slider(
        "Línea vertical derecha (%)",
        min_value=int(st.session_state["x1_pct"]) + 20,
        max_value=90,
        step=5,
        key="x2_pct",
    )
with right:
    st.slider("Línea horizontal superior (%)", 10, 70, step=5, key="y1_pct")
    if st.session_state["y2_pct"] < st.session_state["y1_pct"] + 20:
        st.session_state["y2_pct"] = min(90, st.session_state["y1_pct"] + 20)
    st.slider(
        "Línea horizontal inferior (%)",
        min_value=int(st.session_state["y1_pct"]) + 20,
        max_value=90,
        step=5,
        key="y2_pct",
    )

positions = (
    int(st.session_state["x1_pct"]),
    int(st.session_state["x2_pct"]),
    int(st.session_state["y1_pct"]),
    int(st.session_state["y2_pct"]),
)
means, _, _ = cell_means(
    gradient_integral,
    analysis_image.width,
    analysis_image.height,
    *positions,
)
center_gradient = means[1][1]
outer_gradients = [means[r][c] for r in range(3) for c in range(3) if (r, c) != (1, 1)]
outer_mean = float(np.mean(outer_gradients))

st.subheader("Resultados")
view1, view2 = st.columns(2)
with view1:
    st.markdown("**Imagen con cuadrícula**")
    st.image(draw_grid(original, positions), use_container_width=True)
with view2:
    st.markdown("**Mapa de magnitud del gradiente**")
    st.image(draw_grid(make_gradient_preview(gradient), positions), use_container_width=True)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Gradiente medio de la celda central", f"{center_gradient:.2f}")
metric2.metric("Gradiente medio de las 8 celdas externas", f"{outer_mean:.2f}")
metric3.metric("Contraste local de gradiente", f"{center_gradient - outer_mean:.2f}")

st.markdown("**Gradiente medio por celda** — valores más altos indican cambios de intensidad más rápidos.")
labels = ["Izquierda", "Centro", "Derecha"]
rows = []
for row in range(3):
    rows.append({
        "Fila": ["Superior", "Central", "Inferior"][row],
        "Izquierda": round(means[row][0], 3),
        "Centro": round(means[row][1], 3),
        "Derecha": round(means[row][2], 3),
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

with st.expander("¿Cómo se selecciona la zona?"):
    st.latex(r"G(x,y)=\sqrt{I_x(x,y)^2+I_y(x,y)^2}")
    st.write(
        "Primero se calcula el gradiente de la imagen en escala de grises. Después se evalúan distintas posiciones "
        "de la cuadrícula y se calcula el gradiente medio de cada celda. La búsqueda favorece la celda central "
        "cuando tiene un gradiente alto frente a las ocho celdas que la rodean:"
    )
    st.latex(r"J=\overline{G}_{\mathrm{central}}-0.35\,\overline{G}_{\mathrm{exterior}}")
    st.write(
        "Es un criterio heurístico para localizar bordes y texturas; no equivale necesariamente a la zona más importante "
        "para una persona. El ruido también puede producir gradientes altos, por eso puedes ajustar el suavizado previo."
    )
