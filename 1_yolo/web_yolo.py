# app.py
# ---------------------------------------------------------
# OSA Visual (Streamlit) - UX simple y clara
# - Sube una foto
# - Ingresa datos: id, local, distrito, lat, lon, productos esperados (+ puntaje google opcional)
# - YOLO cuenta "productos disponibles"
# - Calcula OSA = (disponibles / esperados) * 100
# - Muestra imagen anotada y tabla final
# - Guarda/actualiza un Excel en la carpeta indicada
# ---------------------------------------------------------

import os
import io
import time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from ultralytics import YOLO

# ---------- Config UI ----------
st.set_page_config(page_title="OSA Visual", page_icon="🛒", layout="wide")

# Pequeño tuneo estético
st.markdown("""
<style>
    .main .block-container {padding-top: 1rem; padding-bottom: 2rem; }
    .st-emotion-cache-16idsys p {font-size: 0.95rem;}
    .metric {text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------- Parámetros por defecto (ajústalos si quieres) ----------
MODEL_PATH_DEFAULT = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/1_yolo/yolo11n.pt"
SAVE_DIR_DEFAULT   = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/bronze"  # <- tu ruta
EXCEL_NAME         = "osa_resultados.xlsx"  # nombre del archivo excel

# Solo contar esta(s) clase(s) por nombre según model.names
TARGET_CLASS_NAMES = {"bottle"}

# ---------- Caché de modelo ----------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)

# ---------- Utilidades ----------
def bytes_to_bgr(image_bytes: bytes):
    """Convierte bytes -> BGR (cv2)"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)           # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, image  # BGR (para anotación) y PIL (para otros usos)

def annotate_and_count(model, bgr_image: np.ndarray, conf_thr: float, target_class_names=None):
    """
    Ejecuta YOLO y cuenta SOLO las clases deseadas (por nombre), p.ej. {"bottle"}.
    Evita evaluar tensores en contexto booleano.
    """
    # ---- names dicts ----
    names = model.names
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}

    # ---- target ids por nombre ----
    target_ids = None
    lower_targets = set()
    if target_class_names:
        lower_targets = {t.lower() for t in target_class_names}
        found_ids = [name_to_id[t] for t in lower_targets if t in name_to_id]
        target_ids = found_ids if len(found_ids) > 0 else None

    # ---- inferencia (si target_ids existe, filtramos en el forward) ----
    results = model(bgr_image, conf=conf_thr, classes=target_ids)
    res = results[0]

    # ---- extraer clases detectadas sin usar "or []" ----
    classes = []
    if res.boxes is not None and getattr(res.boxes, "cls", None) is not None:
        # cls es un tensor -> pásalo a lista de ints
        classes = res.boxes.cls.int().cpu().tolist()

    # ---- filtro post (si no filtramos en el forward y pidieron clases objetivo) ----
    if target_ids is None and lower_targets and classes:
        keep_idx = [i for i, c in enumerate(classes)
                    if id_to_name.get(int(c), "").lower() in lower_targets]
        classes = [classes[i] for i in keep_idx]
        if keep_idx:
            res.boxes = res.boxes[keep_idx]   # deja solo las cajas objetivo
        else:
            # deja sin cajas si ninguna coincide; res.plot() dibuja la imagen limpia
            res.boxes = res.boxes[:0]

    # ---- conteo ----
    from collections import Counter
    conteo_por_clase = Counter(classes)
    total = int(sum(conteo_por_clase.values()))

    # ---- anotación ----
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, total, conteo_por_clase, id_to_name

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_or_append_excel(df_row: pd.DataFrame, folder: str, filename: str):
    ensure_dir(folder)
    fpath = os.path.join(folder, filename)
    if os.path.exists(fpath):
        # lee existente, concat, y sobrescribe
        old = pd.read_excel(fpath)
        out = pd.concat([old, df_row], ignore_index=True)
        out.to_excel(fpath, index=False)
    else:
        df_row.to_excel(fpath, index=False)
    return fpath

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    model_path = st.text_input("Ruta del modelo YOLO", value=MODEL_PATH_DEFAULT)
    conf_thr   = st.slider("Umbral de confianza", 0.05, 0.90, 0.20, 0.05)
    save_dir   = st.text_input("Carpeta para guardar Excel", value=SAVE_DIR_DEFAULT)
    st.caption("Tip: si usas Colab/Drive la ruta suele ser `/content/drive/MyDrive/...`")

# ---------- Header ----------
st.title("🛒 OSA Visual — On-Shelf Availability")
st.markdown("Sube una foto del estante, llena los datos y obtén **productos disponibles** y **OSA** al instante.")

# ---------- Formulario ----------
with st.form(key="osa-form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        id_ = st.text_input("ID (*)", placeholder="Ejem: 000123")
        local = st.text_input("Local (*)", placeholder="Ejem: Tda San Borja")
    with col2:
        distrito = st.text_input("Distrito (*)", placeholder="Ejem: San Borja")
        puntaje_google = st.number_input("Puntaje Google (opcional)", min_value=0.0, max_value=5.0, step=0.1, value=None, placeholder="0.0–5.0")
    with col3:
        lat = st.number_input("Latitud (*)", format="%.6f")
        lon = st.number_input("Longitud (*)", format="%.6f")

    expected = st.number_input("Productos esperados (*)", min_value=0, step=1)
    uploaded = st.file_uploader("Sube la foto (png/jpg/jpeg) (*)", type=["png","jpg","jpeg"])

    submitted = st.form_submit_button("Procesar", use_container_width=True)

# ---------- Lógica ----------
if submitted:
    # Validaciones mínimas
    missing = []
    if not id_: missing.append("ID")
    if not local: missing.append("Local")
    if not distrito: missing.append("Distrito")
    if lat is None: missing.append("Latitud")
    if lon is None: missing.append("Longitud")
    if expected is None: missing.append("Productos esperados")
    if not uploaded: missing.append("Imagen")
    if missing:
        st.error("Faltan campos obligatorios: " + ", ".join(missing))
        st.stop()

    # Carga modelo (cacheado)
    with st.spinner("Cargando modelo YOLO..."):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"No se pudo cargar el modelo en: `{model_path}`\nError: {e}")
            st.stop()

    # Procesar imagen
    with st.spinner("Analizando imagen..."):
        try:
            bgr_img, pil_img = bytes_to_bgr(uploaded.read())
            annotated_rgb, total_found, conteo_clase, class_names = annotate_and_count(
    model, bgr_img, conf_thr, TARGET_CLASS_NAMES)
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")
            st.stop()

    # Calcular OSA
    if expected == 0:
        osa = 0.0
    else:
        osa = (total_found / expected) * 100.0

    # Panel de resultados (izq: imagen / der: métricas + tabla)
    left, right = st.columns([3,2], gap="large")

    with left:
        st.subheader("📷 Imagen anotada")
        st.image(annotated_rgb, use_container_width=True, caption=f"Detecciones: {total_found} (umbral={conf_thr:.2f})")

        with st.expander("Ver conteo por clase"):
            if conteo_clase:
                df_cl = pd.DataFrame(
                    [{"clase_id": k, "clase": class_names.get(k, str(k)), "conteo": v}
                     for k, v in sorted(conteo_clase.items(), key=lambda x: -x[1])]
                )
                st.dataframe(df_cl, hide_index=True, use_container_width=True)
            else:
                st.info("Sin detecciones sobre el umbral.")

    with right:
        st.subheader("📊 Métricas")
        c1, c2, c3 = st.columns(3)
        c1.metric("Disponibles", f"{total_found}")
        c2.metric("Esperados", f"{expected}")
        c3.metric("OSA %", f"{osa:.2f}%")

        st.subheader("🧾 Registro")
        fila = {
            "id": id_,
            "local": local,
            "distrito": distrito,
            "puntaje google": puntaje_google if puntaje_google is not None else "",
            "latitud": float(lat),
            "longitud": float(lon),
            "productos disponibles": int(total_found),
            "productos esperados": int(expected),
            "OSA": round(osa, 2),
            "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        df_row = pd.DataFrame([fila], columns=[
            "id","local","distrito","puntaje google","latitud","longitud",
            "productos disponibles","productos esperados","OSA","fecha_hora"
        ])
        st.dataframe(df_row.drop(columns=["fecha_hora"]), hide_index=True, use_container_width=True)

        # Vista rápida en mapa
        st.map(df_row.rename(columns={"latitud":"lat","longitud":"lon"})[["lat","lon"]])

        # Guardar/actualizar Excel
        try:
            path_saved = save_or_append_excel(df_row.drop(columns=["fecha_hora"]), save_dir, EXCEL_NAME)
            st.success(f"Guardado/actualizado: {path_saved}")
        except Exception as e:
            st.error(f"No se pudo guardar el Excel en `{save_dir}`: {e}")

