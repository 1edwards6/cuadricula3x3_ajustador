import io
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

st.set_page_config(page_title="Analizador de cuadrícula 3×3", page_icon="🧩", layout="wide")

st.title("Analizador automático de cuadrícula 3×3")
st.write(
    "Busca automáticamente dos cortes verticales y dos horizontales para que la celda central "
    "tenga una desviación estándar de luminancia alta, mientras que las ocho celdas periféricas "
    "tengan una desviación menor. No hay parámetros de optimización que ajustar."
)

st.info(
    "Criterio usado: maximizar la desviación estándar de la celda central menos el promedio "
    "de las desviaciones estándar de las otras ocho celdas. Para evitar regiones diminutas, "
    "cada franja debe ocupar al menos el 10 % de la dimensión correspondiente. "
    "Las posiciones candidatas se prueban cada 5 %."
)

uploaded = st.file_uploader(
    "Carga una imagen para analizar",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
)


def integral_image(array: np.ndarray) -> np.ndarray:
    """Tabla integral con una fila y columna iniciales de ceros."""
    return np.pad(
        np.cumsum(np.cumsum(array.astype(np.float64), axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )


def rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    return float(
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def rect_std(
    integral: np.ndarray,
    integral_sq: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    n = width * height
    mean = rect_sum(integral, x0, y0, x1, y1) / n
    mean_sq = rect_sum(integral_sq, x0, y0, x1, y1) / n
    variance = max(0.0, mean_sq - mean * mean)
    return float(np.sqrt(variance))


def cut_pairs(length: int):
    """Cortes del 10 % al 90 %, en pasos de 5 %, dejando franjas >= 10 %."""
    percentages = range(10, 91, 5)
    positions = sorted({int(round(length * p / 100.0)) for p in percentages})
    return [
        (a, b)
        for a, b in combinations(positions, 2)
        if a >= 0 and b < length and (b - a) >= int(np.ceil(length * 0.10))
        and a >= int(np.ceil(length * 0.10))
        and (length - b) >= int(np.ceil(length * 0.10))
    ]


def analyse_grid(luminance: np.ndarray):
    h, w = luminance.shape
    int_y = integral_image(luminance)
    int_y2 = integral_image(luminance * luminance)

    xs = cut_pairs(w)
    ys = cut_pairs(h)
    if not xs or not ys:
        raise ValueError("La imagen es demasiado pequeña para probar la cuadrícula con seguridad.")

    best = None
    best_key = None

    # La tabla integral permite calcular la desviación de cualquier celda en O(1).
    for y1, y2 in ys:
        ybounds = (0, y1, y2, h)
        for x1, x2 in xs:
            xbounds = (0, x1, x2, w)
            stds = []
            for row in range(3):
                for col in range(3):
                    stds.append(
                        rect_std(
                            int_y,
                            int_y2,
                            xbounds[col],
                            ybounds[row],
                            xbounds[col + 1],
                            ybounds[row + 1],
                        )
                    )
            center_std = stds[4]
            outer_stds = [value for i, value in enumerate(stds) if i != 4]
            outer_mean = float(np.mean(outer_stds))
            score = center_std - outer_mean
            # En caso de empate, preferir centro más variable y periferia menos variable.
            key = (score, center_std, -outer_mean)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "stds": stds,
                    "center_std": center_std,
                    "outer_mean": outer_mean,
                    "score": score,
                }

    return best


def draw_grid(image: Image.Image, result: dict) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    # Contraste doble: contorno oscuro y línea amarilla para que se vea sobre casi cualquier imagen.
    for xy in [(result["x1"], 0, result["x1"], h), (result["x2"], 0, result["x2"], h),
               (0, result["y1"], w, result["y1"]), (0, result["y2"], w, result["y2"])]:
        draw.line(xy, fill=(0, 0, 0), width=max(3, w // 350))
        draw.line(xy, fill=(255, 220, 0), width=max(1, w // 700))

    # Resaltar la celda central sin cubrir su contenido.
    width = max(3, w // 250)
    draw.rectangle(
        [result["x1"], result["y1"], result["x2"], result["y2"]],
        outline=(255, 40, 40),
        width=width,
    )
    return annotated


if uploaded is None:
    st.markdown("### ¿Qué buscará el algoritmo?")
    st.markdown(
        "La cuadrícula se coloca para que la **celda central (2,2)** destaque por su desviación "
        "estándar de luminancia, en comparación con las ocho celdas restantes. "
        "El resultado es automático y se puede revisar con la tabla de métricas."
    )
else:
    try:
        raw = Image.open(io.BytesIO(uploaded.getvalue()))
        raw = ImageOps.exif_transpose(raw).convert("RGB")
        original_size = raw.size

        # Redimensionar solo para el análisis y la visualización; conserva la proporción.
        max_side = 900
        analysis_image = raw.copy()
        analysis_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        rgb = np.asarray(analysis_image, dtype=np.float64)
        # Luminancia perceptual aproximada en la escala original de 0 a 255.
        luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

        with st.spinner("Buscando la mejor posición de la cuadrícula…"):
            result = analyse_grid(luminance)

        h, w = luminance.shape
        x1, x2, y1, y2 = result["x1"], result["x2"], result["y1"], result["y2"]
        x_percentages = [100 * x1 / w, 100 * x2 / w]
        y_percentages = [100 * y1 / h, 100 * y2 / h]

        # Métricas detalladas de las nueve celdas.
        xbounds = [0, x1, x2, w]
        ybounds = [0, y1, y2, h]
        labels = [
            "(1,1) Superior izquierda", "(1,2) Superior centro", "(1,3) Superior derecha",
            "(2,1) Centro izquierda", "(2,2) Centro", "(2,3) Centro derecha",
            "(3,1) Inferior izquierda", "(3,2) Inferior centro", "(3,3) Inferior derecha",
        ]
        data = []
        for row in range(3):
            for col in range(3):
                x0, xend = xbounds[col], xbounds[col + 1]
                y0, yend = ybounds[row], ybounds[row + 1]
                idx = row * 3 + col
                area_pct = 100 * ((xend - x0) * (yend - y0)) / (w * h)
                data.append({
                    "Celda": labels[idx],
                    "Ancho (px)": xend - x0,
                    "Alto (px)": yend - y0,
                    "Área (%)": area_pct,
                    "Desviación estándar": result["stds"][idx],
                })
        metrics_df = pd.DataFrame(data)

        left, right = st.columns([1.35, 1])
        with left:
            st.subheader("Cuadrícula encontrada")
            st.image(draw_grid(analysis_image, result), caption="Cuadrícula automática; la celda central está resaltada en rojo.", use_container_width=True)
            st.caption(
                f"Imagen original: {original_size[0]} × {original_size[1]} px. "
                f"Cálculos sobre una versión de {w} × {h} px para mantener el análisis ágil."
            )
        with right:
            st.subheader("Posición de los ejes")
            st.metric("Eje vertical izquierdo", f"{x_percentages[0]:.1f} % del ancho")
            st.metric("Eje vertical derecho", f"{x_percentages[1]:.1f} % del ancho")
            st.metric("Eje horizontal superior", f"{y_percentages[0]:.1f} % del alto")
            st.metric("Eje horizontal inferior", f"{y_percentages[1]:.1f} % del alto")

            st.subheader("Objetivo de selección")
            a, b = st.columns(2)
            a.metric("Desviación del centro", f"{result['center_std']:.2f}")
            b.metric("Promedio periférico", f"{result['outer_mean']:.2f}")
            st.metric("Centro − periferia", f"{result['score']:.2f}")

        st.subheader("Desviación por celda")
        chart_df = metrics_df.set_index("Celda")[["Desviación estándar"]]
        st.bar_chart(chart_df)

        with st.expander("Ver métricas de las nueve celdas"):
            st.dataframe(
                metrics_df.style.format({"Área (%)": "{:.2f}", "Desviación estándar": "{:.3f}"}),
                hide_index=True,
                use_container_width=True,
            )

        st.download_button(
            "Descargar métricas en CSV",
            data=metrics_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="metricas_cuadricula_3x3.csv",
            mime="text/csv",
        )

        st.caption(
            "Interpretación: una desviación estándar alta indica mayor variabilidad de luminancia, "
            "pero no garantiza por sí sola que la región tenga más detalle perceptual. La posición "
            "encontrada es la mejor entre las posiciones candidatas evaluadas, no necesariamente "
            "el óptimo continuo exacto."
        )
    except Exception as exc:
        st.error(f"No fue posible analizar la imagen: {exc}")
