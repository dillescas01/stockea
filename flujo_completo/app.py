# app.py
# =====================================================================
# Streamlit app unificada:
#  1) OSA Visual (YOLO + histórico bronze)
#  2) Forecast 7 días (ExtraTrees + histórico + forecast => silver)
#  3) Ruteo (usa por defecto el silver generado, o un Excel subido)
# =====================================================================

import os
import io
import math
import json
import re
import requests  
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st
from ultralytics import YOLO
from sklearn.ensemble import ExtraTreesRegressor
from stores_meta import STORE_META
import pydeck as pdk

# =========================
# CONFIG GENERAL / RUTAS
# =========================

st.set_page_config(
    page_title="OSA + Forecast + Ruteo",
    page_icon="🛒",
    layout="wide"
)

# --- Ajusta estas rutas a tu entorno ---
BASE_DATA_DIR = r"C:\Users\illes\Desktop\stockea\data"
BRONZE_DIR_DEFAULT = os.path.join(BASE_DATA_DIR, "1_bronze")
SILVER_DIR_DEFAULT = os.path.join(BASE_DATA_DIR, "2_silver")

# Excel donde se acumula el histórico OSA (bronze)
OSA_EXCEL_NAME = "osa_hist_Tambo_UTEC.xlsx"
BRONZE_EXCEL_DEFAULT = os.path.join(BRONZE_DIR_DEFAULT, OSA_EXCEL_NAME)

# Carpeta de forecast dentro de bronze
FORECAST_DIR_DEFAULT = os.path.join(BRONZE_DIR_DEFAULT, "forecast")
FORECAST_XLSX_NAME = "forecast_UTEC_7d_best.xlsx"

# Archivo SILVER por defecto (histórico + forecast)
SILVER_HISTORY_DEFAULT = os.path.join(
    SILVER_DIR_DEFAULT,
    "osa_hist_Tambo_UTEC_with_forecast.xlsx"
)

# Ruta modelo YOLO por defecto
MODEL_PATH_DEFAULT = r"C:\Users\illes\Desktop\stockea\1_yolo\yolo11n.pt"

# Clases YOLO a contar
TARGET_CLASS_NAMES = {"bottle"}

# Parámetros forecast
H = 7            # horizonte (7 días)
SEED = 42

# Parámetros ruteo (depot por defecto)
DEFAULT_DEPOT_LAT = -12.068071984223696
DEFAULT_DEPOT_LON = -76.94734607980992
ESTRATO_WEIGHT_DEFAULT = {"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}

# Endpoint OSRM para ruteo realista (puedes cambiar a tu propio servidor si quieres)
OSRM_BASE_URL = "https://router.project-osrm.org"

# =========================
# UTILIDADES GENERALES
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_or_append_excel(df_row: pd.DataFrame, folder: str, filename: str):
    ensure_dir(folder)
    fpath = os.path.join(folder, filename)
    if os.path.exists(fpath):
        old = pd.read_excel(fpath)
        out = pd.concat([old, df_row], ignore_index=True)
        out.to_excel(fpath, index=False)
    else:
        df_row.to_excel(fpath, index=False)
    return fpath

# =========================
# 1) OSA VISUAL (YOLO)
# =========================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)

def bytes_to_bgr(image_bytes: bytes):
    """Convierte bytes -> BGR (cv2) + PIL."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)           # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, image

def annotate_and_count(model, bgr_image: np.ndarray, conf_thr: float, target_class_names=None):
    """
    Ejecuta YOLO y cuenta SOLO las clases deseadas (por nombre), p.ej. {"bottle"}.
    Evita evaluar tensores en contexto booleano.
    """
    names = model.names
    if isinstance(names, dict):
        id_to_name = {int(k): str(v) for k, v in names.items()}
    else:
        id_to_name = {i: str(v) for i, v in enumerate(names)}
    name_to_id = {v.lower(): k for k, v in id_to_name.items()}

    target_ids = None
    lower_targets = set()
    if target_class_names:
        lower_targets = {t.lower() for t in target_class_names}
        found_ids = [name_to_id[t] for t in lower_targets if t in name_to_id]
        target_ids = found_ids if len(found_ids) > 0 else None

    # Inference (si tenemos target_ids, filtramos en el forward)
    results = model(bgr_image, conf=conf_thr, classes=target_ids)
    res = results[0]

    classes = []
    if res.boxes is not None and getattr(res.boxes, "cls", None) is not None:
        classes = res.boxes.cls.int().cpu().tolist()

    # Filtro post si no filtramos en el forward
    if target_ids is None and lower_targets and classes:
        keep_idx = [
            i for i, c in enumerate(classes)
            if id_to_name.get(int(c), "").lower() in lower_targets
        ]
        classes = [classes[i] for i in keep_idx]
        if keep_idx:
            res.boxes = res.boxes[keep_idx]
        else:
            res.boxes = res.boxes[:0]

    conteo_por_clase = Counter(classes)
    total = int(sum(conteo_por_clase.values()))

    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, total, conteo_por_clase, id_to_name



def page_osa_visual():
    st.header("🛒 Paso 1 — OSA Visual (detección con YOLO)")

    # --- Configuración en sidebar ---
    with st.sidebar:
        st.subheader("Config OSA Visual")
        model_path = st.text_input("Ruta modelo YOLO", value=MODEL_PATH_DEFAULT)
        conf_thr = st.slider("Umbral confianza", 0.05, 0.90, 0.20, 0.05)
        save_dir = st.text_input("Carpeta histórico (BRONZE)", value=BRONZE_DIR_DEFAULT)
        excel_name = st.text_input("Nombre Excel histórico", value=OSA_EXCEL_NAME)

    st.markdown("""
Selecciona el **local**, ingresa los **productos esperados** (puedes tomar de la base sugerida) 
y sube la **foto del estante**. El resto de datos se completan automáticamente desde el catálogo.
""")

    # --- Selección de tienda (catálogo) ---
    # Ordenamos por id para que sea estable
    store_ids = sorted(STORE_META.keys())
    default_store_id = "TUB0001" if "TUB0001" in store_ids else store_ids[0]

    selected_store_id = st.selectbox(
        "Selecciona el local",
        options=store_ids,
        index=store_ids.index(default_store_id) if default_store_id in store_ids else 0,
        format_func=lambda sid: f"{sid} — {STORE_META[sid]['local']}"
    )

    store_info = STORE_META[selected_store_id]

    # Panel con info precargada
    with st.expander("Ver detalle del local (precargado)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**ID:** {store_info['id']}")
            st.write(f"**Local:** {store_info['local']}")
            st.write(f"**Distrito:** {store_info['distrito']}")
            st.write(f"**Estrato:** {store_info['estrato']}")
        with c2:
            st.write(f"**Puntaje Google:** {store_info['puntaje_google']}")
            st.write(f"**Latitud:** {store_info['latitud']}")
            st.write(f"**Longitud:** {store_info['longitud']}")
            st.write(f"**Esperados base sugeridos:** {store_info['esperados_base']}")

    # --- Formulario simple: solo esperados + imagen ---
    with st.form(key="osa-form"):
        expected = st.number_input(
            "Productos esperados (*)",
            min_value=0,
            step=1,
            value=int(store_info["esperados_base"]),
            help="Puedes usar el valor base sugerido o ajustarlo."
        )

        uploaded = st.file_uploader(
            "Sube la foto del estante (png/jpg/jpeg) (*)",
            type=["png", "jpg", "jpeg"]
        )

        submitted = st.form_submit_button("Procesar y guardar", use_container_width=True)

    if not submitted:
        st.info("Selecciona el local, ajusta los productos esperados y sube la foto. Luego haz clic en **Procesar y guardar**.")
        return

    # --- Validaciones mínimas ---
    if uploaded is None:
        st.error("Falta subir la imagen.")
        return

    # --- Cargar modelo YOLO ---
    with st.spinner("Cargando modelo YOLO..."):
        try:
            model = load_model(model_path)
        except Exception as e:
            st.error(f"No se pudo cargar el modelo en `{model_path}`.\nError: {e}")
            return

    # --- Procesar imagen ---
    with st.spinner("Analizando imagen..."):
        try:
            bgr_img, _ = bytes_to_bgr(uploaded.read())
            annotated_rgb, total_found, conteo_clase, class_names = annotate_and_count(
                model, bgr_img, conf_thr, TARGET_CLASS_NAMES
            )
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")
            return

    # --- Calcular OSA ---
    osa = 0.0 if expected == 0 else (total_found / expected) * 100.0

    left, right = st.columns([3, 2], gap="large")

    # --- Imagen anotada y conteo ---
    with left:
        st.subheader("📷 Imagen anotada")
        st.image(
            annotated_rgb,
            use_container_width=True,
            caption=f"Detecciones: {total_found} (umbral={conf_thr:.2f})"
        )

        with st.expander("Ver conteo por clase"):
            if conteo_clase:
                df_cl = pd.DataFrame(
                    [
                        {"clase_id": k, "clase": class_names.get(k, str(k)), "conteo": v}
                        for k, v in sorted(conteo_clase.items(), key=lambda x: -x[1])
                    ]
                )
                st.dataframe(df_cl, hide_index=True, use_container_width=True)
            else:
                st.info("Sin detecciones sobre el umbral.")

    # --- Métricas + fila registrada ---
    with right:
        st.subheader("📊 Métricas")
        c1, c2, c3 = st.columns(3)
        c1.metric("Disponibles", f"{total_found}")
        c2.metric("Esperados", f"{expected}")
        c3.metric("OSA %", f"{osa:.2f}%")

        st.subheader("🧾 Registro generado (fila nueva)")

        fila = {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "id": store_info["id"],
            "local": store_info["local"],
            "distrito": store_info["distrito"],
            "puntaje google": float(store_info["puntaje_google"]),
            "latitud": float(store_info["latitud"]),
            "longitud": float(store_info["longitud"]),
            "estrato": store_info.get("estrato", ""),  # la guardamos también
            "productos disponibles": int(total_found),
            "productos esperados": int(expected),
            "osa": round(osa, 2),
        }

        df_row = pd.DataFrame([fila])
        st.dataframe(df_row, hide_index=True, use_container_width=True)

        # Mapa rápido
        st.map(df_row.rename(columns={"latitud": "lat", "longitud": "lon"})[["lat", "lon"]])

        # Guardar en histórico BRONZE
        try:
            path_saved = save_or_append_excel(df_row, save_dir, excel_name)
            st.success(f"Guardado/actualizado histórico en: `{path_saved}`")
            st.session_state["bronze_path"] = path_saved
        except Exception as e:
            st.error(f"No se pudo guardar el Excel en `{save_dir}`: {e}")


# =========================
# 2) FORECAST 7 DÍAS
# =========================

def parse_fecha(x):
    """Convierte serial Excel o string a datetime; si ya es fecha, la devuelve."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x)
    s = str(x).strip().replace(",", "")
    try:
        n = int(float(s))
        if 30000 <= n <= 60000:  # 1990–2100 aprox
            return pd.to_datetime(n, unit="D", origin="1899-12-30")
        return pd.to_datetime(s, errors="coerce")
    except:
        return pd.to_datetime(s, errors="coerce")
    


def load_clean_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "fecha" not in df.columns:
        raise ValueError("No se encontró columna 'fecha' en el archivo.")

    # ===== Manejo de la fecha (serial Excel o datetime) =====
    col = df["fecha"]

    # Caso 1: ya viene como datetime -> la usamos directo
    if pd.api.types.is_datetime64_any_dtype(col):
        df["fecha"] = col

    # Caso 2: viene como número (45_292, 45_293, ...) -> serial Excel clásico
    elif pd.api.types.is_numeric_dtype(col):
        df["fecha"] = pd.to_datetime(col, unit="D", origin="1899-12-30", errors="coerce")

    # Caso 3: viene como string tipo "45,292" o "2024-01-01"
    else:
        # quitamos comas tipo "45,292" -> "45292"
        s = col.astype(str).str.replace(",", "", regex=False)
        # primero intentamos como número serial de Excel
        ser_num = pd.to_numeric(s, errors="coerce")
        mask_num = ser_num.notna()
        fechas = pd.to_datetime(ser_num[mask_num], unit="D", origin="1899-12-30", errors="coerce")
        # el resto, lo intentamos parsear como fecha normal
        fechas_rest = pd.to_datetime(s[~mask_num], errors="coerce")
        df["fecha"] = pd.concat([fechas, fechas_rest]).sort_index()

    # Quitamos filas sin fecha válida y ordenamos
    df = df.dropna(subset=["fecha"])
    df = df.sort_values("fecha")

    # Una fila por día (nos quedamos con la ÚLTIMA por fecha)
    df = df.drop_duplicates(subset="fecha", keep="last")

    # Índice por fecha con frecuencia diaria
    df = df.set_index("fecha").asfreq("D")

    # ===== Dejar solo columnas relevantes =====
    KEEP_COLS = [
        "id", "local", "distrito", "puntaje google",
        "latitud", "longitud",
        "productos disponibles", "productos esperados",
        "osa", "estrato"
    ]
    keep_cols_present = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep_cols_present]

    # numéricos base
    for c in ["productos esperados", "productos disponibles", "osa",
              "puntaje google", "latitud", "longitud"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # imputaciones mínimas
    if "productos esperados" in df.columns:
        df["productos esperados"] = df["productos esperados"].ffill()
    else:
        raise ValueError("Falta columna 'productos esperados'.")

    if "productos disponibles" in df.columns:
        df["productos disponibles"] = df["productos disponibles"].fillna(0)
    else:
        raise ValueError("Falta columna 'productos disponibles'.")

    # Recalcular OSA limpio (para histórico)
    df["osa"] = (df["productos disponibles"] / df["productos esperados"] * 100)\
        .replace([np.inf, -np.inf], np.nan)

    # día de semana
    df["dow"] = df.index.dayofweek

    return df




def build_supervised(df: pd.DataFrame, h: int = H):
    """
    Arma features y targets para forecast multi-step.
    Importante: OSA y sus lags NO entran como input al modelo.
    Solo usamos:
      - productos disponibles / esperados
      - codificación de día de semana
      - lags y rolling de disponibles/esperados
    """
    data = df.copy()

    # quitar columnas que NO se usan en el modelo
    drop_cols = ["id", "local", "distrito", "puntaje google",
                 "latitud", "longitud", "estrato"]
    for c in drop_cols:
        if c in data.columns:
            data = data.drop(columns=c)

    # asegurar numéricos base
    for c in ["productos disponibles", "productos esperados"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # codificación cíclica del día de semana
    if "dow" not in data.columns:
        data["dow"] = data.index.dayofweek
    data["dow_sin"] = np.sin(2*np.pi*data["dow"]/7.0)
    data["dow_cos"] = np.cos(2*np.pi*data["dow"]/7.0)

    # lags de productos disponibles
    for L in [1, 2, 3, 7, 14]:
        data[f"disp_lag{L}"] = data["productos disponibles"].shift(L)

    # lags de productos esperados
    for L in [1, 7, 14]:
        data[f"esp_lag{L}"] = data["productos esperados"].shift(L)

    # rolling stats
    data["disp_roll7_mean"] = data["productos disponibles"].rolling(7).mean()
    data["disp_roll7_std"]  = data["productos disponibles"].rolling(7).std()
    data["esp_roll7_mean"]  = data["productos esperados"].rolling(7).mean()

    # targets multi-step: productos disponibles futuros t+1..t+H
    Y_cols = []
    for k in range(1, h+1):
        col_y = f"y_tplus{k}"
        data[col_y] = data["productos disponibles"].shift(-k)
        Y_cols.append(col_y)

    # columnas de entrada (features) EXPLÍCITAS
    X_cols = [
        "dow_sin", "dow_cos",
        "disp_lag1", "disp_lag2", "disp_lag3", "disp_lag7", "disp_lag14",
        "esp_lag1", "esp_lag7", "esp_lag14",
        "disp_roll7_mean", "disp_roll7_std", "esp_roll7_mean",
    ]

    # Nos quedamos solo con features + targets para evitar columnas raras
    cols_for_sup = X_cols + Y_cols
    data_sup = data[cols_for_sup]

    # Drop de filas con NaN en alguna de las columnas usadas
    sup = data_sup.dropna().copy()

    if sup.empty:
        raise ValueError("Después de construir lags/rollings no queda ningún dato válido (sup vacío). Revisa el histórico.")

    X_all = sup[X_cols].astype(float)
    Y_all = sup[Y_cols].astype(float)

    return X_all, Y_all, X_cols, sup



def forecast_next_7d(df: pd.DataFrame, X_cols: list[str], estimator) -> pd.DataFrame:
    """
    Genera forecast 7 días. OSA_pred se calcula solo con prod_disp_pred / prod_esp_ref.
    """
    data = df.copy()
    drop_cols = ["id", "local", "distrito", "puntaje google", "latitud", "longitud"]
    for c in drop_cols:
        if c in data.columns:
            data = data.drop(columns=c)

    for c in ["productos disponibles", "productos esperados"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    if "dow" not in data.columns:
        data["dow"] = data.index.dayofweek
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7.0)

    for L in [1, 2, 3, 7, 14]:
        data[f"disp_lag{L}"] = data["productos disponibles"].shift(L)
    for L in [1, 7, 14]:
        data[f"esp_lag{L}"] = data["productos esperados"].shift(L)

    data["disp_roll7_mean"] = data["productos disponibles"].rolling(7).mean()
    data["disp_roll7_std"] = data["productos disponibles"].rolling(7).std()
    data["esp_roll7_mean"] = data["productos esperados"].rolling(7).mean()

    t0 = data.index.max()
    X_last = data.loc[[t0], X_cols].astype(float)
    yhat = estimator.predict(X_last)[0].astype(float)

    future_dates = pd.date_range(t0 + pd.Timedelta(days=1), periods=H, freq="D")

    exp_future = []
    for d in future_dates:
        ref = d - pd.Timedelta(days=7)
        if ref in df.index and not pd.isna(df.loc[ref, "productos esperados"]):
            exp_future.append(int(df.loc[ref, "productos esperados"]))
        else:
            exp_future.append(int(df["productos esperados"].iloc[-1]))
    exp_future = np.array(exp_future, dtype=int)

    yhat_int = np.rint(yhat).astype(int)
    yhat_int = np.clip(yhat_int, 0, exp_future)

    osa_pred = np.round((yhat_int / np.maximum(exp_future, 1)) * 100.0, 2)

    out = pd.DataFrame({
        "fecha": future_dates,
        "prod_esp_ref": exp_future,
        "prod_disp_pred": yhat_int,
        "osa_pred": osa_pred
    })
    return out

def _next_ids(last_id: str, n: int):
    """Genera n IDs consecutivos si el último sigue patrón prefijo+digitos."""
    if not isinstance(last_id, str):
        return None
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", last_id.strip())
    if not m:
        return None
    prefix, num = m.group(1), int(m.group(2))
    width = len(m.group(2))
    return [f"{prefix}{str(num + i).zfill(width)}" for i in range(1, n + 1)]

def append_forecast_to_history(base_df: pd.DataFrame,
                               forecast_df: pd.DataFrame,
                               silver_path: str) -> pd.DataFrame:
    """
    Acumula predicciones al histórico en archivo 'silver'.
    Si existe, lo actualiza sin duplicar fechas; si no, parte del histórico base_df.
    """
    if os.path.exists(silver_path):
        silver_df = pd.read_excel(silver_path)
        silver_df.columns = [c.strip().lower() for c in silver_df.columns]
        silver_df["fecha"] = silver_df["fecha"].apply(parse_fecha)
        silver_df = silver_df.sort_values("fecha").reset_index(drop=True)
    else:
        silver_df = base_df.reset_index().copy()
        silver_df = silver_df.reset_index(drop=True)

    const_vals = {}
    for c in ["local", "distrito", "latitud", "longitud", "puntaje google"]:
        if c in silver_df.columns and silver_df[c].notna().any():
            const_vals[c] = silver_df[c].dropna().iloc[-1]
        elif c in base_df.columns and base_df[c].notna().any():
            const_vals[c] = base_df[c].dropna().iloc[-1]
        else:
            const_vals[c] = np.nan

    new_ids = None
    if "id" in silver_df.columns and silver_df["id"].notna().any():
        last_id_val = silver_df["id"].dropna().iloc[-1]
        new_ids = _next_ids(str(last_id_val), len(forecast_df))

    rows = []
    for i, row in forecast_df.iterrows():
        r = {
            "fecha": row["fecha"],
            "productos disponibles": int(row["prod_disp_pred"]),
            "productos esperados": int(row["prod_esp_ref"]),
            "osa": float(row["osa_pred"]),
        }
        for k, v in const_vals.items():
            r[k] = v
        if new_ids is not None:
            r["id"] = new_ids[len(rows)]
        rows.append(r)
    append_df = pd.DataFrame(rows)

    first_fc_date = forecast_df["fecha"].min()
    silver_df = silver_df[silver_df["fecha"] < first_fc_date].copy()

    out_all = pd.concat([silver_df, append_df], ignore_index=True, sort=False)
    out_all = out_all.sort_values("fecha").reset_index(drop=True)

    preferred = [
        "fecha", "id", "local", "distrito", "puntaje google", "latitud", "longitud",
        "productos disponibles", "productos esperados", "osa"
    ]
    others = [c for c in out_all.columns if c not in preferred]
    cols = [c for c in preferred if c in out_all.columns] + others
    out_all = out_all[cols]

    ensure_dir(os.path.dirname(silver_path))
    out_all.to_excel(silver_path, index=False)
    return out_all

def page_forecast():
    st.header("📈 Paso 2 — Forecast 7 días")

    bronze_default = st.session_state.get("bronze_path", BRONZE_EXCEL_DEFAULT)

    col1, col2 = st.columns(2)
    with col1:
        input_path = st.text_input("Archivo histórico (BRONZE)", value=bronze_default)
    with col2:
        silver_path = st.text_input("Archivo SILVER (histórico + forecast)", value=SILVER_HISTORY_DEFAULT)

    forecast_dir = st.text_input("Carpeta para guardar forecast (solo predicciones)", value=FORECAST_DIR_DEFAULT)

    run_fc = st.button("Calcular forecast 7 días")

    if not run_fc:
        st.info("Configura rutas y presiona **Calcular forecast 7 días**.")
        return

    try:
        with st.spinner("Cargando y limpiando histórico..."):
            df = load_clean_df(input_path)
        st.success(f"Histórico cargado. Rango: {df.index.min().date()} → {df.index.max().date()} ({len(df)} días)")

        with st.spinner("Construyendo dataset supervisado..."):
            X_all, Y_all, X_cols, _ = build_supervised(df, h=H)

        with st.spinner("Entrenando ExtraTrees y generando forecast..."):
            est = ExtraTreesRegressor(
                n_estimators=800,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=SEED,
                n_jobs=-1
            )
            est.fit(X_all, Y_all)
            forecast_df = forecast_next_7d(df, X_cols, est)

        # Guardar solo predicciones
        ensure_dir(forecast_dir)
        forecast_path = os.path.join(forecast_dir, FORECAST_XLSX_NAME)
        forecast_df.to_excel(forecast_path, index=False)

        # Acumular en SILVER
        hist_plus_fc = append_forecast_to_history(df, forecast_df, silver_path)

        st.session_state["forecast_df"] = forecast_df
        st.session_state["silver_df"] = hist_plus_fc

        st.success(f"Forecast guardado en: `{forecast_path}`")
        st.success(f"Histórico + forecast guardado/actualizado en: `{silver_path}`")

        st.subheader("📅 Forecast 7 días")
        st.dataframe(forecast_df, use_container_width=True)

        st.subheader("📈 Últimos días + forecast (vista rápida)")
        tail_hist = df.tail(14).reset_index()
        tail_hist["tipo"] = "hist"
        fc_plot = forecast_df.copy()
        fc_plot["tipo"] = "fc"
        fc_plot = fc_plot.rename(columns={
            "fecha": "fecha",
            "prod_disp_pred": "productos disponibles",
            "prod_esp_ref": "productos esperados",
            "osa_pred": "osa"
        })
        df_plot = pd.concat([tail_hist[["fecha", "productos disponibles", "tipo"]],
                             fc_plot[["fecha", "productos disponibles", "tipo"]]])

        st.line_chart(
            df_plot.pivot(index="fecha", columns="tipo", values="productos disponibles"),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error al ejecutar el forecast: {e}")

# =========================
# 3) RUTEO INTELIGENTE
# =========================

def osrm_route_segment(lat1, lon1, lat2, lon2):
    """
    Pide a OSRM la ruta en auto entre (lat1, lon1) y (lat2, lon2).
    Devuelve: distancia_km, tiempo_min, geometría (lista de [lon, lat]).
    Si falla, cae a línea recta (haversine).
    """
    try:
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = f"{OSRM_BASE_URL}/route/v1/driving/{coords}"
        params = {
            "overview": "full",
            "geometries": "geojson"
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        route = data["routes"][0]
        dist_km = route["distance"] / 1000.0
        time_min = route["duration"] / 60.0
        geom = route["geometry"]["coordinates"]  # lista de [lon, lat]
        return dist_km, time_min, geom
    except Exception:
        # Fallback: línea recta si algo falla
        d = haversine_km(lat1, lon1, lat2, lon2)
        return d, None, [[lon1, lat1], [lon2, lat2]]
    
def build_osrm_path(points, route):
    """
    Dados:
      - points: [(lat, lon)] con índice 0 = depot, 1..n = tiendas
      - route: lista de índices de tiendas en el orden a visitar (ej. [2,1,3])
    Construye:
      - path_coords: geometría continua Depot -> t1 -> t2 -> ... -> Depot (lista de [lon, lat])
      - total_dist_km: suma de distancias de todos los tramos (km)
      - total_time_min: suma de tiempos (min, puede ser None si OSRM falla)
      - segments_info: lista de dicts con info tramo a tramo
    """
    path_coords = []
    total_dist_km = 0.0
    total_time_min = 0.0
    any_time = False
    segments_info = []

    seq = [0] + route + [0]  # empezamos y terminamos en depot

    for i in range(len(seq) - 1):
        a = seq[i]
        b = seq[i+1]
        lat1, lon1 = points[a]
        lat2, lon2 = points[b]

        dist_km, time_min, geom = osrm_route_segment(lat1, lon1, lat2, lon2)
        total_dist_km += dist_km
        if time_min is not None:
            total_time_min += time_min
            any_time = True

        segments_info.append({
            "from_index": a,
            "to_index": b,
            "from_lat": lat1,
            "from_lon": lon1,
            "to_lat": lat2,
            "to_lon": lon2,
            "distance_km": dist_km,
            "time_min": time_min
        })

        # concatenamos geometría, evitando repetir el primer punto de cada tramo
        if not path_coords:
            path_coords.extend(geom)
        else:
            path_coords.extend(geom[1:])

    return path_coords, total_dist_km, (total_time_min if any_time else None), segments_info


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_distance_matrix(points):
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 0.0 if i == j else haversine_km(points[i][0], points[i][1],
                                                      points[j][0], points[j][1])
    return D

def parse_fecha_col(fecha_series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(fecha_series):
        return fecha_series.copy()
    s = fecha_series.copy()
    try:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            return pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
    except Exception:
        pass
    return pd.to_datetime(s, errors="coerce")

def compute_priority(df_day: pd.DataFrame,
                     w_osa: float, w_estr: float, w_gap: float,
                     estrato_weight_map: dict):
    osa_priority = 1.0 - (pd.to_numeric(df_day["osa"], errors="coerce") / 100.0)
    estr_priority = df_day["estrato"].astype(str).str.upper().map(estrato_weight_map).fillna(0.25).values
    gap_vals = (pd.to_numeric(df_day["productos esperados"], errors="coerce") -
                pd.to_numeric(df_day["productos disponibles"], errors="coerce")).astype(float).values
    gap_norm = (gap_vals / gap_vals.max()) if np.nanmax(gap_vals) > 0 else np.zeros_like(gap_vals)
    priority = w_osa * osa_priority.values + w_estr * estr_priority + w_gap * gap_norm

    pr_min, pr_max = float(np.nanmin(priority)), float(np.nanmax(priority))
    if pr_max - pr_min > 1e-9:
        priority_norm = (priority - pr_min) / (pr_max - pr_min)
    else:
        priority_norm = np.zeros_like(priority)
    return priority, priority_norm

def solve_route(df_day: pd.DataFrame,
                depot_lat: float, depot_lon: float,
                priority: np.ndarray,
                lambda_prior: float,
                alpha_heuristic: float):
    points = [(depot_lat, depot_lon)] + list(zip(pd.to_numeric(df_day["latitud"], errors="coerce"),
                                                 pd.to_numeric(df_day["longitud"], errors="coerce")))
    D = build_distance_matrix(points)
    n = len(points)
    C = list(range(1, n))

    route = None
    method_pretty = None
    obj_val = None

    try:
        import pyomo.environ as pyo

        m = pyo.ConcreteModel()
        m.N = pyo.Set(initialize=list(range(n)))
        m.C = pyo.Set(initialize=C)

        m.x = pyo.Var(m.N, m.N, domain=pyo.Binary)
        m.u = pyo.Var(m.C, bounds=(1, len(C)), domain=pyo.Reals)

        def no_self(m, i):
            return m.x[i, i] == 0
        m.no_self = pyo.Constraint(m.N, rule=no_self)

        def outdeg(m, i):
            return sum(m.x[i, j] for j in m.N if j != i) == 1
        def indeg(m, j):
            return sum(m.x[i, j] for i in m.N if i != j) == 1
        m.outdeg = pyo.Constraint(m.N, rule=outdeg)
        m.indeg = pyo.Constraint(m.N, rule=indeg)

        M = len(C)
        def mtz(m, i, j):
            if i != j:
                return m.u[i] - m.u[j] + M * m.x[i, j] <= M - 1
            return pyo.Constraint.Skip
        m.mtz = pyo.Constraint(m.C, m.C, rule=mtz)

        def obj(m):
            dist = sum(D[i, j] * m.x[i, j] for i in m.N for j in m.N)
            prio = sum(float(priority[i-1]) * m.u[i] for i in m.C)
            return dist + lambda_prior * prio
        m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

        solver = None
        for cand in ("glpk", "cbc"):
            s = pyo.SolverFactory(cand)
            if s is not None and s.available():
                solver = s
                break
        if solver is None:
            raise RuntimeError("No solver disponible")

        solver.solve(m, tee=False)
        method_pretty = "Óptimo (solver)"

        succ = {}
        for i in range(n):
            for j in range(n):
                if i != j and pyo.value(m.x[i, j]) > 0.5:
                    succ[i] = j
                    break

        seq = [0]
        while True:
            nxt = succ[seq[-1]]
            seq.append(nxt)
            if nxt == 0:
                break
        route = [i for i in seq[1:-1]]
        obj_val = pyo.value(m.obj)

    except Exception:
        method_pretty = "Heurístico (rápido)"
        remaining = C.copy()
        current = 0
        route = []
        while remaining:
            nxt = min(remaining, key=lambda k: D[current, k] + alpha_heuristic * priority[k-1])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt

        def route_score(rt):
            dist = D[0, rt[0]] + sum(D[rt[i], rt[i+1]] for i in range(len(rt)-1)) + D[rt[-1], 0]
            pen = sum(priority[rt[i]-1] * (i+1) for i in range(len(rt)))
            return dist + 1.0 * pen

        improved = True
        while improved:
            improved = False
            best = route_score(route)
            for i in range(1, len(route)-1):
                for k in range(i+1, len(route)):
                    new_rt = route[:i] + route[i:k+1][::-1] + route[k+1:]
                    sc = route_score(new_rt)
                    if sc + 1e-9 < best:
                        route = new_rt
                        best = sc
                        improved = True

    dist_total = 0.0
    if route:
        dist_total = D[0, route[0]] + \
                     sum(D[route[i], route[i+1]] for i in range(len(route)-1)) + \
                     D[route[-1], 0]

    return route, dist_total, method_pretty, obj_val, points, D

def color_from_priority(p_norm: float):
    """0 -> verde; 1 -> rojo."""
    r = int(34 + (220 - 34) * p_norm)
    g = int(139 + (20 - 139) * p_norm)
    b = int(34 + (60 - 34) * p_norm)
    return [r, g, b, 220]

def build_route_table(df_day: pd.DataFrame, route: list, priority: np.ndarray):
    tbl = pd.DataFrame({
        "orden": range(1, len(route)+1),
        "id": [df_day.iloc[i-1]["id"] for i in route],
        "local": [df_day.iloc[i-1]["local"] for i in route],
        "distrito": [df_day.iloc[i-1]["distrito"] for i in route],
        "estrato": [df_day.iloc[i-1]["estrato"] for i in route],
        "osa": [df_day.iloc[i-1]["osa"] for i in route],
        "prioridad": [round(priority[i-1], 3) for i in route],
        "lat": [df_day.iloc[i-1]["latitud"] for i in route],
        "lon": [df_day.iloc[i-1]["longitud"] for i in route],
    })
    return tbl



def build_map(points, route, df_day, priority_norm, path_coords, depot_name="Depot"):
    rows = []
    # depot
    rows.append({
        "name": depot_name,
        "lon": points[0][1], "lat": points[0][0],
        "orden": 0,
        "radius": 120,
        "color": [0, 153, 0, 240],
        "label": "Depot"
    })
    # tiendas
    for idx, node in enumerate(route, start=1):
        lat, lon = points[node]
        pnorm = float(priority_norm[node-1])
        color = color_from_priority(pnorm)
        radius = 70 + 80 * pnorm
        label = f"{idx}"
        rows.append({
            "name": f"{df_day.iloc[node-1]['id']} — {df_day.iloc[node-1]['local']}",
            "lon": lon, "lat": lat,
            "orden": idx,
            "radius": radius,
            "color": color,
            "label": label
        })
    dots = pd.DataFrame(rows)

    # Geometría real de la ruta (ya procesada con OSRM)
    path_df = pd.DataFrame({"path": [path_coords]})

    scatter = pdk.Layer(
        "ScatterplotLayer",
        dots,
        get_position='[lon, lat]',
        get_radius="radius",
        get_fill_color="color",
        pickable=True
    )
    text = pdk.Layer(
        "TextLayer",
        dots,
        get_position='[lon, lat]',
        get_text='label',
        get_size=16,
        get_color=[0, 0, 0],
        get_alignment_baseline='"top"'
    )
    path_layer = pdk.Layer(
        "PathLayer",
        path_df,
        get_path="path",
        width_scale=5,
        width_min_pixels=4,
        get_width=4,
        get_color=[60, 60, 60, 180]
    )

    lat_center = float(np.mean([p[0] for p in points]))
    lon_center = float(np.mean([p[1] for p in points]))
    view_state = pdk.ViewState(
        latitude=lat_center,
        longitude=lon_center,
        zoom=12.5,
        pitch=0,
        bearing=0
    )

    deck = pdk.Deck(
        layers=[path_layer, scatter, text],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="light"
    )
    return deck



def page_ruteo():
    st.header("🚚 Paso 3 — Ruteo inteligente")

    st.write("Regla: menor **OSA**, mayor **vulnerabilidad (D>C>B>A)** y mayor **brecha** (esperados − disponibles) ⇒ se atienden antes.")

    # Fuente de datos: session_state (silver) o archivo subido
    df = None
    used_session = False

    if "silver_df" in st.session_state:
        used_session = True
        df = st.session_state["silver_df"].copy()
        st.success("Usando histórico + forecast generado en el Paso 2.")
    up = st.file_uploader("Opcional: subir Excel de tiendas (sobrescribe el generado)", type=["xlsx"])
    if up is not None:
        try:
            df = pd.read_excel(up)
            used_session = False
            st.info("Usando Excel subido para el ruteo.")
        except Exception as e:
            st.error(f"No pude leer el Excel subido: {e}")
            return

    if df is None:
        st.info("Primero genera un forecast (Paso 2) o sube un Excel con tiendas.")
        return

    REQUIRED_COLS = [
        "fecha", "id", "local", "distrito", "puntaje google", "latitud", "longitud",
        "productos disponibles", "productos esperados", "osa"
    ]

    df.columns = [c.strip().lower() for c in df.columns]

    missing_base = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_base:
        st.error(f"Faltan columnas base en el DataFrame: {missing_base}")
        return

    # Estrato: si no existe, creamos una por defecto
    if "estrato" not in df.columns:
        estrato_default = st.selectbox("Estrato por defecto para las tiendas (si no existe columna)", ["A", "B", "C", "D"], index=2)
        df["estrato"] = estrato_default

    # Tipos
    for c in ["puntaje google", "latitud", "longitud", "productos disponibles", "productos esperados", "osa"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_fecha_dt"] = parse_fecha_col(df["fecha"])
    df = df.sort_values(["_fecha_dt", "id"]).reset_index(drop=True)

    dias = df["_fecha_dt"].dropna().dt.date.unique().tolist()
    if not dias:
        dias_serial = sorted(df["fecha"].unique().tolist())
        day_choice = st.selectbox("Elige el día (serial Excel)", dias_serial)
        df_day = df[df["fecha"] == day_choice].copy()
        st.write(f"**Día (serial):** {day_choice}")
    else:
        day_choice = st.selectbox("Elige el día a rutear", dias)
        df_day = df[df["_fecha_dt"].dt.date == day_choice].copy()
        st.write(f"**Día seleccionado:** {day_choice}")

    st.subheader("Tiendas del día seleccionado")
    st.dataframe(
        df_day[["id", "local", "distrito", "latitud", "longitud",
                "productos disponibles", "productos esperados", "osa", "estrato"]],
        use_container_width=True
    )

    with st.sidebar:
        st.subheader("Parámetros de ruteo")
        depot_lat = st.number_input("Depot lat", value=DEFAULT_DEPOT_LAT, format="%.8f")
        depot_lon = st.number_input("Depot lon", value=DEFAULT_DEPOT_LON, format="%.8f")

        st.subheader("Pesos de prioridad")
        w_osa = st.slider("Peso OSA", 0.0, 1.0, 0.60, 0.05)
        w_estr = st.slider("Peso Estrato", 0.0, 1.0, 0.30, 0.05)
        w_gap = st.slider("Peso Brecha", 0.0, 1.0, 0.10, 0.05)

        st.subheader("Trade-off / Heurístico")
        lambda_prior = st.slider("λ (prioridad vs km)", 0.0, 5.0, 1.0, 0.1)
        alpha_heur = st.slider("alpha (heurístico)", 0.0, 20.0, 5.0, 0.5)

        run_btn = st.button("Calcular ruta")

    if not run_btn:
        return

    if df_day.empty:
        st.warning("No hay tiendas para el día seleccionado.")
        return

    total_w = max(1e-9, w_osa + w_estr + w_gap)
    w_osa_n, w_estr_n, w_gap_n = w_osa/total_w, w_estr/total_w, w_gap/total_w

    priority, pnorm = compute_priority(df_day, w_osa_n, w_estr_n, w_gap_n, ESTRATO_WEIGHT_DEFAULT)
    route, dist_km, method_pretty, obj_val, points, _ = solve_route(
        df_day, depot_lat, depot_lon, priority, lambda_prior, alpha_heur
    )
    path_coords, real_dist_km, real_time_min, segments = build_osrm_path(points, route)

    if not route:
        st.warning("No se encontró una ruta (¿hay tiendas válidas?)")
        return

    tbl = build_route_table(df_day, route, priority)
    st.subheader("Ruta sugerida (orden y prioridad)")
    st.dataframe(tbl, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Paradas", len(route))
    c2.metric("Distancia total (km, ruta real)", f"{real_dist_km:.2f}")
    if real_time_min is not None:
        c3.metric("Tiempo estimado (min)", f"{real_time_min:.1f}")
    else:
        c3.metric("Tiempo estimado (min)", "N/D")

    with st.expander("Ver detalle tramo a tramo (distancia y tiempo)", expanded=False):
        seg_rows = []
        for seg in segments:
            a = seg["from_index"]
            b = seg["to_index"]

            if a == 0:
                from_name = "Depot"
            else:
                ra = df_day.iloc[a-1]
                from_name = f"{ra['id']} — {ra['local']}"

            if b == 0:
                to_name = "Depot"
            else:
                rb = df_day.iloc[b-1]
                to_name = f"{rb['id']} — {rb['local']}"

            seg_rows.append({
                "Desde": from_name,
                "Hasta": to_name,
                "Distancia (km)": round(seg["distance_km"], 3),
                "Tiempo (min)": round(seg["time_min"], 1) if seg["time_min"] is not None else None,
                "Desde (lat, lon)": f"{seg['from_lat']:.5f}, {seg['from_lon']:.5f}",
                "Hasta (lat, lon)": f"{seg['to_lat']:.5f}, {seg['to_lon']:.5f}",
            })

        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

    st.markdown("""
**Leyenda de colores:**  
- **Verde** = baja prioridad (menos urgencia)  
- **Amarillo/Naranja** = prioridad media  
- **Rojo** = alta prioridad (más urgente)  
- **Círculo más grande** = mayor prioridad  
**Números** sobre cada punto = orden de visita.
""")
    



    deck = build_map(points, route, df_day, pnorm, path_coords, depot_name="Depot (ArcaContinental)")
    st.pydeck_chart(deck, use_container_width=True)

    out_csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar ruta (CSV)", out_csv,
                       file_name="ruta_sugerida.csv", mime="text/csv")

# =========================
# NAV / MAIN
# =========================

st.markdown("""
<style>
    .main .block-container {padding-top: 0.5rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navegación",
    (
        "1️⃣ OSA Visual",
        "2️⃣ Forecast 7 días",
        "3️⃣ Ruteo"
    )
)

if page.startswith("1"):
    page_osa_visual()
elif page.startswith("2"):
    page_forecast()
else:
    page_ruteo()