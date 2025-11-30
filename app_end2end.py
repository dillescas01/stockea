# app_end2end.py
# ===================================================
# Streamlit END-TO-END:
# 1) Merchand selecciona tienda + sube foto
# 2) YOLO cuenta productos -> OSA
# 3) Actualiza bronze + histórico UTEC
# 4) Ejecuta forecast 7d (utec_forecast_benchmark.run_forecast)
# 5) Genera gold_tiendas_7d (build_gold_tiendas_7d.build_gold_tiendas)
# 6) Calcula y muestra ruta óptima / heurística en mapa
# ===================================================

import os
import io
import math
from datetime import datetime, date

import altair as alt
import graphviz

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
import pydeck as pdk
from ultralytics import YOLO

# Intenta importar los metadatos de la tienda
try:
    from flujo_completo.stores_meta import STORE_META
except ImportError:
    st.error("Error fatal: No se encontró el archivo 'stores_meta.py'.")
    st.stop()

# === RUTAS BASE (ajusta si hace falta) ===
BASE_DIR = "/Users/johar/Desktop/Desarrollo/proyecto_dpd"
BRONZE_OSA = os.path.join(BASE_DIR, "data/bronze/osa_resultados.xlsx")
UTEC_HIST  = os.path.join(BASE_DIR, "data/1_bronze/osa_hist_Tambo_UTEC.xlsx")
SILVER_UTEC = os.path.join(BASE_DIR, "data/2_silver/osa_hist_Tambo_UTEC_with_forecast.xlsx")
GOLD_PATH  = os.path.join(BASE_DIR, "data/3_gold/gold_tiendas_7d.xlsx")
YOLO_MODEL_DEFAULT = os.path.join(BASE_DIR, "1_yolo/yolo11n.pt")

# === Importar tus scripts como módulos ===
# (Asegúrate que utec_forecast_benchmark.py y build_gold_tiendas_7d.py
# estén en la misma carpeta que este script, o en tu PYTHONPATH)
try:
    import utec_forecast_benchmark 
    import build_gold_tiendas_7d 
except ImportError:
    st.error("Error: No se pudieron importar los scripts 'utec_forecast_benchmark.py' o 'build_gold_tiendas_7d.py'. Asegúrate de que estén en la misma carpeta.")
    st.stop()


# ========= Utils generales =========
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def bytes_to_bgr(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, image

def annotate_and_count(model, bgr_image: np.ndarray, conf_thr: float, target_class_names=None):
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

    results = model(bgr_image, conf=conf_thr, classes=target_ids)
    res = results[0]

    classes = []
    if res.boxes is not None and getattr(res.boxes, "cls", None) is not None:
        classes = res.boxes.cls.int().cpu().tolist()

    if target_ids is None and lower_targets and classes:
        keep_idx = [i for i, c in enumerate(classes)
                    if id_to_name.get(int(c), "").lower() in lower_targets]
        classes = [classes[i] for i in keep_idx]
        if keep_idx:
            res.boxes = res.boxes[keep_idx]
        else:
            res.boxes = res.boxes[:0]

    from collections import Counter
    conteo_por_clase = Counter(classes)
    total = int(sum(conteo_por_clase.values()))

    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, total, conteo_por_clase, id_to_name

@st.cache_resource(show_spinner=False)
def load_yolo(model_path: str):
    return YOLO(model_path)

def append_capture_to_bronze(capture_row: dict, path: str = BRONZE_OSA):
    ensure_dir(os.path.dirname(path))
    df_new = pd.DataFrame([capture_row])
    if os.path.exists(path):
        old = pd.read_excel(path)
        out = pd.concat([old, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_excel(path, index=False)
    return path

def sync_utec_hist_from_capture(capture_row: dict, hist_path: str = UTEC_HIST):
    """
    Actualiza el histórico diario de UTEC con el conteo real de hoy.
    Si ya existe la fecha -> reemplaza disponibles/esperados/OSA.
    Si no existe -> agrega fila nueva.
    Asume que el archivo histórico ya existe (lo generaste antes con el sintético).
    """
    if not os.path.exists(hist_path):
        # si no existe, simplemente creamos un nuevo histórico con esa fila
        df = pd.DataFrame([capture_row])
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
        df.to_excel(hist_path, index=False)
        return

    df = pd.read_excel(hist_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Manejar columnas duplicadas por si acaso
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # fecha puede venir como date o como string
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    today = pd.to_datetime(capture_row["fecha"]).date()

    mask = df["fecha"] == today
    row_vals = {
        "id": capture_row["id"],
        "local": capture_row["local"],
        "distrito": capture_row["distrito"],
        "puntaje google": capture_row["puntaje google"],
        "latitud": capture_row["latitud"],
        "longitud": capture_row["longitud"],
        "productos disponibles": capture_row["productos disponibles"],
        "productos esperados": capture_row["productos esperados"],
        "OSA": capture_row["OSA"],
    }
    if mask.any():
        for k, v in row_vals.items():
            if k in df.columns:
                df.loc[mask, k] = v
            else:
                df[k] = np.nan
                df.loc[mask, k] = v
    else:
        new_row = {
            "fecha": today,
            **row_vals
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.sort_values("fecha")
    df.to_excel(hist_path, index=False)

# ========= Ruteo utils (resumen de tu app_ruteo) =========
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_distance_matrix(points):
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 0.0 if i == j else haversine_km(points[i][0], points[i][1], points[j][0], points[j][1])
    return D

def compute_priority(df_day: pd.DataFrame,
                     w_osa: float, w_estr: float, w_gap: float,
                     estrato_weight_map: dict):
    osa_priority = 1.0 - (pd.to_numeric(df_day["osa"], errors="coerce") / 100.0)
    estr_priority = df_day["estrato"].astype(str).str.upper().map(estrato_weight_map).fillna(0.25).values
    gap_vals = (pd.to_numeric(df_day["productos esperados"], errors="coerce") -
                pd.to_numeric(df_day["productos disponibles"], errors="coerce")).astype(float).values
    gap_norm = (gap_vals / gap_vals.max()) if np.nanmax(gap_vals) > 0 else np.zeros_like(gap_vals)
    priority = w_osa*osa_priority.values + w_estr*estr_priority + w_gap*gap_norm
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

        def no_self(m, i): return m.x[i, i] == 0
        m.no_self = pyo.Constraint(m.N, rule=no_self)
        def outdeg(m, i): return sum(m.x[i, j] for j in m.N if j != i) == 1
        def indeg(m, j): return sum(m.x[i, j] for i in m.N if i != j) == 1
        m.outdeg = pyo.Constraint(m.N, rule=outdeg)
        m.indeg  = pyo.Constraint(m.N, rule=indeg)

        M = len(C)
        def mtz(m, i, j):
            if i != j:
                return m.u[i] - m.u[j] + M*m.x[i, j] <= M - 1
            return pyo.Constraint.Skip
        m.mtz = pyo.Constraint(m.C, m.C, rule=mtz)

        def obj(m):
            dist = sum(D[i, j]*m.x[i, j] for i in m.N for j in m.N)
            prio = sum(float(priority[i-1])*m.u[i] for i in m.C)
            return dist + lambda_prior*prio
        m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)

        solver = None
        for cand in ("glpk", "cbc"):
            s = pyo.SolverFactory(cand)
            if s is not None and s.available():
                solver = s
                break
        if solver is None:
            raise RuntimeError("No solver")

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
            nxt = min(remaining, key=lambda k: D[current, k] + alpha_heuristic*priority[k-1])
            route.append(nxt)
            remaining.remove(nxt)
            current = nxt

        def route_score(rt):
            dist = D[0, rt[0]] + sum(D[rt[i], rt[i+1]] for i in range(len(rt)-1)) + D[rt[-1], 0]
            pen  = sum(priority[rt[i]-1]*(i+1) for i in range(len(rt)))
            return dist + 1.0*pen

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
                        best  = sc
                        improved = True

    dist_total = 0.0
    if route:
        dist_total = D[0, route[0]] + sum(D[route[i], route[i+1]] for i in range(len(route)-1)) + D[route[-1], 0]
    return route, dist_total, method_pretty, obj_val, points, D

def color_from_priority(p_norm: float):
    r = int(34   + (220-34)  * p_norm)
    g = int(139  + (20-139)  * p_norm)
    b = int(34   + (60-34)   * p_norm)
    return [r, g, b, 220]

def build_map(points, route, df_day, priority_norm, depot_name="Depot (ArcaContinental)"):
    rows = []
    rows.append({
        "name": depot_name,
        "lon": points[0][1], "lat": points[0][0],
        "orden": 0,
        "radius": 120,
        "color": [0, 153, 0, 240],
        "label": "Depot"
    })
    for idx, node in enumerate(route, start=1):
        lat, lon = points[node]
        pnorm = float(priority_norm[node-1])
        color = color_from_priority(pnorm)
        radius = 70 + 80*pnorm
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
    path_coords = [[points[0][1], points[0][0]]] + \
                  [[points[n][1], points[n][0]] for n in route] + \
                  [[points[0][1], points[0][0]]]
    path_df = pd.DataFrame({"path":[path_coords]})

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
    view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=12.5, pitch=0, bearing=0)

    deck = pdk.Deck(
        layers=[path_layer, scatter, text],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="light"
    )
    return deck

def build_route_graph(df_day: pd.DataFrame, route: list[int]) -> graphviz.Digraph:
    """
    Grafo sencillo:
    - Nodo 'Depot'
    - Nodos 1..N en orden de visita
    - Flechas Depot -> primera tienda -> ... -> última tienda -> Depot
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")  # izquierda a derecha

    # Nodo depot
    dot.node("Depot", "Depot\n(ArcaContinental)", shape="box", style="filled", color="#cccccc")

    # Nodos de tiendas
    for idx, node in enumerate(route, start=1):
        row = df_day.iloc[node-1]
        label = f"{idx}. {row['id']}\n{row['local']} ({row['distrito']})"
        dot.node(str(node), label, shape="circle")

    # Arcos
    if route:
        # Depot -> primera tienda
        dot.edge("Depot", str(route[0]), label="start")
        # Camino interno
        for i in range(len(route)-1):
            dot.edge(str(route[i]), str(route[i+1]))
        # Última tienda -> Depot
        dot.edge(str(route[-1]), "Depot", label="end")

    return dot

# ========= UI =========
st.set_page_config(page_title="OSA + Forecast + Ruteo (end-to-end)", layout="wide")
st.title("🛒🚚 OSA end-to-end — Conteo, Forecast y Ruteo")

with st.sidebar:
    st.header("⚙️ Configuración YOLO")
    model_path = st.text_input("Ruta modelo YOLO", value=YOLO_MODEL_DEFAULT)
    conf_thr   = st.slider("Umbral de confianza", 0.05, 0.90, 0.20, 0.05)
    target_class = st.text_input("Clase a contar", value="bottle")
    st.markdown("---")
    st.header("⚙️ Ruteo")
    depot_lat = st.number_input("Depot lat", value=-12.068071984223696, format="%.8f")
    depot_lon = st.number_input("Depot lon", value=-76.94734607980992, format="%.8f")
    w_osa  = st.slider("Peso OSA",   0.0, 1.0, 0.60, 0.05)
    w_estr = st.slider("Peso Estrato",0.0, 1.0, 0.30, 0.05)
    w_gap  = st.slider("Peso Brecha", 0.0, 1.0, 0.10, 0.05)
    lambda_prior = st.slider("λ (prioridad vs km)", 0.0, 5.0, 1.0, 0.1)
    alpha_heur   = st.slider("alpha (heurístico)", 0.0, 20.0, 5.0, 0.5)

# ===== Paso 1: Selección de tienda + foto =====
st.subheader("1️⃣ Selecciona tienda y sube foto")

store_keys = []
for sid, meta in STORE_META.items():
    label = f"{sid} — {meta['local']} ({meta['distrito']})"
    store_keys.append((label, sid))
label_to_id = {lbl: sid for lbl, sid in store_keys}
choice = st.selectbox("Tienda", [lbl for lbl, _ in store_keys])
store_id = label_to_id[choice]
meta = STORE_META[store_id]

colA, colB, colC = st.columns(3)
with colA:
    st.metric("ID", meta["id"])
    st.metric("Local", meta["local"])
with colB:
    st.metric("Distrito", meta["distrito"])
    st.metric("Puntaje Google", meta["puntaje_google"])
with colC:
    st.metric("Latitud", f"{meta['latitud']:.6f}")
    st.metric("Longitud", f"{meta['longitud']:.6f}")

expected_today = st.number_input(
    "Productos esperados hoy",
    min_value=0,
    step=1,
    value=int(meta["esperados_base"])
)

uploaded = st.file_uploader("Sube la foto del estante (png/jpg/jpeg)", type=["png","jpg","jpeg"])
run_all = st.button("🚀 Procesar TODO (YOLO + histórico + forecast + ruteo)", use_container_width=True)

if not run_all:
    st.stop()

if uploaded is None:
    st.error("Sube una imagen primero.")
    st.stop()

# ===== Paso 2: YOLO =====
with st.spinner("Cargando modelo YOLO..."):
    try:
        model = load_yolo(model_path)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()

with st.spinner("Analizando imagen..."):
    try:
        bgr_img, pil_img = bytes_to_bgr(uploaded.read())
        annotated_rgb, total_found, conteo_clase, class_names = annotate_and_count(
            model, bgr_img, conf_thr, {target_class}
        )
    except Exception as e:
        st.error(f"No se pudo procesar la imagen: {e}")
        st.stop()

if expected_today == 0:
    osa = 0.0
else:
    osa = (total_found / expected_today) * 100.0

left, right = st.columns([3, 2])

with left:
    st.subheader("📷 Imagen anotada")
    st.image(annotated_rgb, use_container_width=True,
             caption=f"Detecciones: {total_found} ({target_class}), umbral={conf_thr:.2f}")

with right:
    st.subheader("📊 OSA calculada")
    c1, c2, c3 = st.columns(3)
    c1.metric("Disponibles", f"{total_found}")
    c2.metric("Esperados", f"{expected_today}")
    c3.metric("OSA %", f"{osa:.2f}%")

today_date = date.today()
capture_row = {
    "fecha": today_date,
    "id": meta["id"],
    "local": meta["local"],
    "distrito": meta["distrito"],
    "puntaje google": meta["puntaje_google"],
    "latitud": meta["latitud"],
    "longitud": meta["longitud"],
    "productos disponibles": int(total_found),
    "productos esperados": int(expected_today),
    "OSA": round(osa, 2),
}

st.subheader("🧾 Registro a guardar")
st.dataframe(pd.DataFrame([capture_row]), hide_index=True, use_container_width=True)

# ===== Paso 3: Guardar en bronze + histórico =====
with st.spinner("Guardando en bronze y actualizando histórico..."):
    bronze_path = append_capture_to_bronze(capture_row, BRONZE_OSA)
    st.success(f"Actualizado bronze: {bronze_path}")

    if store_id == "TUB0001":
        sync_utec_hist_from_capture(capture_row, UTEC_HIST)
        st.success(f"Histórico UTEC actualizado: {UTEC_HIST}")
    else:
        st.info("Esta tienda no alimenta aún el histórico de forecast (solo UTEC).")

# ===== Paso 4: Forecast + Silver + Gold =====
if store_id == "TUB0001":
    try:
        with st.spinner("Corriendo forecast 7d (utec_forecast_benchmark)..."):
            # ==========================================================
            # CORRECCIÓN: Tu función se llama run_forecast()
            # ==========================================================
            utec_forecast_benchmark.run_forecast()
            st.success(f"Silver actualizado: {SILVER_UTEC}")

        with st.spinner("Generando GOLD (gold_tiendas_7d.xlsx)..."):
            # ==========================================================
            # CORRECCIÓN: Tu función se llama build_gold_tiendas()
            # ==========================================================
            build_gold_tiendas_7d.build_gold_tiendas()
            st.success(f"Gold actualizado: {GOLD_PATH}")
    
    except Exception as e:
        st.error(f"Falló la ejecución del forecast o del build_gold: {e}")
        st.exception(e) # Muestra el traceback completo
        st.stop()


    # === Visualización del histórico + forecast (desde Silver) ===
    try:
        hist = pd.read_excel(SILVER_UTEC) # Leer de SILVER, no UTEC_HIST
        hist.columns = [c.strip().lower() for c in hist.columns]
        
        # Manejar columnas duplicadas por si acaso
        if hist.columns.duplicated().any():
            hist = hist.loc[:, ~hist.columns.duplicated(keep='first')]
            
        hist["fecha"] = pd.to_datetime(hist["fecha"])
        hist = hist.sort_values("fecha")

        # Últimos 60 días de historia + 7 de forecast
        hist_tail = hist.tail(60 + 7).copy()
        
        # Identificar las filas de forecast (donde 'productos disponibles' es NaN
        # y 'prod_disp_pred' existe en tu script)
        hist_tail['tipo'] = np.where(hist_tail['productos disponibles'].isna(), 'Forecast', 'Real')

        st.subheader("📈 Histórico UTEC + Forecast 7d (Silver)")

        base = alt.Chart(hist_tail).encode(
            x=alt.X("fecha:T", title="Fecha")
        )
        
        # Línea de OSA Real (donde tipo = 'Real')
        line_real = base.mark_line(point=True).transform_filter(
            alt.datum.tipo == 'Real'
        ).encode(
            y=alt.Y("osa:Q", title="OSA (%)", scale=alt.Scale(zero=False)),
            color=alt.value("#1f77b4"), # Azul
            tooltip=["fecha:T", "osa:Q", "productos disponibles:Q", "productos esperados:Q"]
        )
        
        # Línea de Forecast (donde tipo = 'Forecast')
        line_forecast = base.mark_line(point=True, strokeDash=[5,5]).transform_filter(
            alt.datum.tipo == 'Forecast'
        ).encode(
            y=alt.Y("osa:Q", title="OSA (%)"),
            color=alt.value("#ff7f0e"), # Naranja
            tooltip=["fecha:T", alt.Tooltip("osa", title="Forecast")]
        )
            
        # Combinar gráficos
        final_chart = alt.layer(line_real, line_forecast).properties(
            title="OSA Real (Azul) vs Forecast (Naranja)"
        ).interactive()
        st.altair_chart(final_chart, use_container_width=True)


        st.caption(
            f"Último dato real ({today_date}): OSA = {osa:.2f}% · "
            f"Disponibles = {total_found} · Esperados = {expected_today}"
        )

    except Exception as e:
        st.warning(f"No se pudo mostrar el histórico/forecast de UTEC: {e}")
    # === FIN GRÁFICO ===

else:
    st.warning("Forecast + gold están basados en UTEC; igual se usará el último GOLD disponible para ruteo.")

# ===== Paso 5: Ruteo desde GOLD =====
if not os.path.exists(GOLD_PATH):
    st.error("No se encontró GOLD para ruteo. Asegúrate de haber corrido el forecast al menos una vez.")
    st.stop()

st.subheader("🚚 Ruteo inteligente (desde GOLD)")

df_gold = pd.read_excel(GOLD_PATH)
df_gold.columns = [c.strip().lower() for c in df_gold.columns]

# Manejar columnas duplicadas por si acaso
if df_gold.columns.duplicated().any():
    df_gold = df_gold.loc[:, ~df_gold.columns.duplicated(keep='first')]

df_gold["fecha_dt"] = pd.to_datetime(df_gold["fecha"], errors="coerce")
df_gold = df_gold.sort_values(["fecha_dt", "id"])

# === CORRECCIÓN: Selección de día para ruteo, filtrando fechas "basura" ===
dias_validas = df_gold["fecha_dt"].dropna()
# Filtrar fechas inválidas (ej. 1969 o 1900)
dias_validas = dias_validas[dias_validas >= pd.to_datetime("2020-01-01")]
dias = sorted(dias_validas.dt.date.unique().tolist())

if not dias:
    st.error("No hay fechas válidas (>= 2020) en GOLD.")
    st.stop()

hoy = date.today()

# Tomamos días desde hoy en adelante (máximo 7)
dias_futuros = [d for d in dias if d >= hoy]
if not dias_futuros:
    # Si no hay futuros, usamos los últimos 7 disponibles
    dias_futuros = dias[-7:]

dias_futuros = dias_futuros[:7]

# --- Caso especial: solo hay 1 día disponible ---
if not dias_futuros:
    st.error("No se encontraron días futuros o recientes para el ruteo.")
    st.stop()
elif len(dias_futuros) == 1:
    day_choice = dias_futuros[0]
    st.info(f"Solo hay un día disponible para ruteo en GOLD: **{day_choice}**")
else:
    offset = st.slider(
        "Día para optimizar ruta (0 = hoy / mínimo disponible)",
        min_value=0,
        max_value=len(dias_futuros) - 1,
        value=0,
        step=1,
    )
    day_choice = dias_futuros[offset]

st.write(f"📅 **Ruta para el día:** {day_choice}")
df_day = df_gold[df_gold["fecha_dt"].dt.date == day_choice].copy()

# === CORRECCIÓN: Eliminar duplicados de tiendas para el día seleccionado ===
df_day = df_day.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

st.write(f"**Tiendas para el {day_choice}:**")
st.dataframe(df_day[["id","local","distrito","latitud","longitud","productos disponibles","productos esperados","osa","estrato"]],
             use_container_width=True)

total_w = max(1e-9, w_osa + w_estr + w_gap)
w_osa_n, w_estr_n, w_gap_n = w_osa/total_w, w_estr/total_w, w_gap/total_w
priority, pnorm = compute_priority(df_day, w_osa_n, w_estr_n, w_gap_n,
                                   {"A":0.10,"B":0.20,"C":0.30,"D":0.40})

route, dist_km, method_pretty, obj_val, points, D = solve_route(
    df_day, depot_lat, depot_lon, priority, lambda_prior, alpha_heur
)

if not route:
    st.warning("No se encontró ruta (¿sin tiendas válidas?)")
    st.stop()

tbl_route = pd.DataFrame({
    "orden": range(1, len(route)+1),
    "id": [df_day.iloc[i-1]["id"] for i in route],
    "local": [df_day.iloc[i-1]["local"] for i in route],
    "distrito": [df_day.iloc[i-1]["distrito"] for i in route],
    "estrato": [df_day.iloc[i-1]["estrato"] for i in route],
    "OSA": [df_day.iloc[i-1]["osa"] for i in route],
    "prioridad": [round(priority[i-1],3) for i in route],
})

st.subheader("Ruta sugerida")
st.dataframe(tbl_route, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Paradas", len(route))
c2.metric("Distancia total (km, ida+vuelta)", f"{dist_km:.2f}")
c3.metric("Método", method_pretty)


if obj_val is not None and method_pretty.startswith("Óptimo"):
    st.caption(f"Valor objetivo: {obj_val:.3f}")

st.markdown("### 🕸️ Grafo de la ruta (nodos y flechas)")
dot = build_route_graph(df_day, route)
st.graphviz_chart(dot)

st.markdown("""
### 🗺️ Mapa de la ruta
**Colores:** - Verde = Depot  
- De verde → rojo = mayor prioridad  
- Círculo más grande = mayor prioridad  
Número dentro del círculo = orden de visita.
""")

deck = build_map(points, route, df_day, pnorm, depot_name="Depot (ArcaContinental)")
st.pydeck_chart(deck, use_container_width=True)

csv_bytes = tbl_route.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar ruta (CSV)", csv_bytes, file_name="ruta_sugerida.csv", mime="text/csv")


if obj_val is not None and method_pretty.startswith("Óptimo"):
    st.caption(f"Valor objetivo: {obj_val:.3f}")

st.markdown("""
**Regla de prioridad:** - Menor **OSA** ⇒ más urgente  
- Estrato más vulnerable (**D > C > B > A**) ⇒ más urgente  
- Mayor **brecha** (esperados − disponibles) ⇒ más urgente  
Pesos ajustables en la barra lateral.
""")