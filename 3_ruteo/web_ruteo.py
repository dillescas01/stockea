# app_ruteo.py
# =========================================
# Streamlit — Ruteo con prioridad (distancia + urgencia)
# Prioridad = 0.6*(1-OSA/100) + 0.3*estrato(D>C>B>A) + 0.1*gap_norm
# Pyomo (GLPK/CBC) -> "Óptimo (solver)"; si no hay solver -> heurístico rápido
# Mapa claro con puntos coloreados por prioridad y numeración del orden.
# =========================================

import math
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---- Config UI ----
st.set_page_config(page_title="Ruteo inteligente (Tambo)", layout="wide")

DEFAULT_DEPOT_LAT = -12.068071984223696
DEFAULT_DEPOT_LON = -76.94734607980992
ESTRATO_WEIGHT_DEFAULT = {"A":0.10, "B":0.20, "C":0.30, "D":0.40}

REQUIRED_COLS = [
    "fecha","id","local","distrito","puntaje google","latitud","longitud",
    "productos disponibles","productos esperados","osa","estrato"
]

# ---- Utilidades geométricas ----
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

# ---- Fechas ----
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

# ---- Prioridad ----
def compute_priority(df_day: pd.DataFrame,
                     w_osa: float, w_estr: float, w_gap: float,
                     estrato_weight_map: dict):
    osa_priority = 1.0 - (pd.to_numeric(df_day["osa"], errors="coerce") / 100.0)  # OSA baja => alta prioridad
    estr_priority = df_day["estrato"].astype(str).str.upper().map(estrato_weight_map).fillna(0.25).values
    gap_vals = (pd.to_numeric(df_day["productos esperados"], errors="coerce") -
                pd.to_numeric(df_day["productos disponibles"], errors="coerce")).astype(float).values
    gap_norm = (gap_vals / gap_vals.max()) if np.nanmax(gap_vals) > 0 else np.zeros_like(gap_vals)
    priority = w_osa*osa_priority.values + w_estr*estr_priority + w_gap*gap_norm
    # Normalizada 0..1 para colores/tamaños
    pr_min, pr_max = float(np.nanmin(priority)), float(np.nanmax(priority))
    if pr_max - pr_min > 1e-9:
        priority_norm = (priority - pr_min) / (pr_max - pr_min)
    else:
        priority_norm = np.zeros_like(priority)
    return priority, priority_norm

# ---- Solver / Heurístico ----
def solve_route(df_day: pd.DataFrame,
                depot_lat: float, depot_lon: float,
                priority: np.ndarray,
                lambda_prior: float,
                alpha_heuristic: float):
    points = [(depot_lat, depot_lon)] + list(zip(pd.to_numeric(df_day["latitud"], errors="coerce"),
                                                 pd.to_numeric(df_day["longitud"], errors="coerce")))
    D = build_distance_matrix(points)
    n = len(points)
    C = list(range(1, n))  # clientes

    route = None
    method_pretty = None
    obj_val = None

    # 1) Intentar Pyomo
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

    # 2) Heurística amigable si no hay solver
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

    # Distancia total (ida y vuelta)
    dist_total = 0.0
    if route:
        dist_total = D[0, route[0]] + sum(D[route[i], route[i+1]] for i in range(len(route)-1)) + D[route[-1], 0]

    return route, dist_total, method_pretty, obj_val, points, D

# ---- Viz helpers ----
def color_from_priority(p_norm: float):
    """0 -> verde; 1 -> rojo (interpolación)"""
    r = int(34   + (220-34)  * p_norm)   # 34..220
    g = int(139  + (20-139)  * p_norm)   # 139..20
    b = int(34   + (60-34)   * p_norm)   # 34..60
    return [r, g, b, 220]

def build_route_table(df_day: pd.DataFrame, route: list, priority: np.ndarray, priority_norm: np.ndarray):
    ids   = [df_day.iloc[i-1]["id"] for i in route]
    names = [df_day.iloc[i-1]["local"] for i in route]
    tbl = pd.DataFrame({
        "orden": range(1, len(route)+1),
        "id": ids,
        "local": names,
        "distrito": [df_day.iloc[i-1]["distrito"] for i in route],
        "estrato":  [df_day.iloc[i-1]["estrato"] for i in route],
        "OSA":      [df_day.iloc[i-1]["osa"] for i in route],
        "prioridad": [round(priority[i-1],3) for i in route],
        "lat":      [df_day.iloc[i-1]["latitud"] for i in route],
        "lon":      [df_day.iloc[i-1]["longitud"] for i in route],
    })
    return tbl

def build_map(points, route, df_day, priority_norm, depot_name="Depot (ArcaContinental)"):
    # Datos de puntos
    rows = []
    # depot
    rows.append({
        "name": depot_name,
        "lon": points[0][1], "lat": points[0][0],
        "orden": 0,
        "radius": 120,
        "color": [0, 153, 0, 240],   # verde brillante
        "label": "Depot"
    })
    # stops
    for idx, node in enumerate(route, start=1):
        lat, lon = points[node]
        pnorm = float(priority_norm[node-1])
        color = color_from_priority(pnorm)
        radius = 70 + 80*pnorm   # más grande si más prioridad
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

    # Path (línea) depot -> stops -> depot
    path_coords = [[points[0][1], points[0][0]]] + \
                  [[points[n][1], points[n][0]] for n in route] + \
                  [[points[0][1], points[0][0]]]
    path_df = pd.DataFrame({"path":[path_coords]})

    # Capa de puntos
    scatter = pdk.Layer(
        "ScatterplotLayer",
        dots,
        get_position='[lon, lat]',
        get_radius="radius",
        get_fill_color="color",
        pickable=True
    )
    # Etiquetas con números grandes de orden
    text = pdk.Layer(
        "TextLayer",
        dots,
        get_position='[lon, lat]',
        get_text='label',
        get_size=16,
        get_color=[0, 0, 0],
        get_alignment_baseline='"top"'
    )
    # Línea de ruta
    path_layer = pdk.Layer(
        "PathLayer",
        path_df,
        get_path="path",
        width_scale=5,
        width_min_pixels=4,
        get_width=4,
        get_color=[60, 60, 60, 180]
    )

    # Vista centrada
    lat_center = float(np.mean([p[0] for p in points]))
    lon_center = float(np.mean([p[1] for p in points]))
    view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=12.5, pitch=0, bearing=0)

    deck = pdk.Deck(
        layers=[path_layer, scatter, text],
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="light"   # <---- mapa claro
    )
    return deck

# ---- UI principal ----
st.title("🚚 Ruteo inteligente — distancia + prioridad")
st.write("**Regla:** menor **OSA**, mayor **vulnerabilidad** (D>C>B>A) y mayor **brecha** (esperados − disponibles) ⇒ **se atienden antes**.")

with st.sidebar:
    st.header("1) Subir archivo GOLD")
    up = st.file_uploader("Sube tu Excel (gold_tiendas_7d.xlsx)", type=["xlsx"])
    st.caption("Columnas requeridas: " + ", ".join(REQUIRED_COLS))

    st.header("2) Parámetros")
    depot_lat = st.number_input("Depot lat", value=DEFAULT_DEPOT_LAT, format="%.8f")
    depot_lon = st.number_input("Depot lon", value=DEFAULT_DEPOT_LON, format="%.8f")

    st.subheader("Pesos de prioridad")
    w_osa  = st.slider("Peso OSA",   0.0, 1.0, 0.60, 0.05)
    w_estr = st.slider("Peso Estrato",0.0, 1.0, 0.30, 0.05)
    w_gap  = st.slider("Peso Brecha", 0.0, 1.0, 0.10, 0.05)

    st.subheader("Trade-off / Heurístico")
    lambda_prior = st.slider("λ (prioridad vs km)", 0.0, 5.0, 1.0, 0.1)
    alpha_heur   = st.slider("alpha (heurístico)", 0.0, 20.0, 5.0, 0.5)

    run_btn = st.button("Calcular ruta")

# ---- Ingesta ----
if up is None:
    st.info("Sube tu Excel `gold_tiendas_7d.xlsx` para continuar.")
    st.stop()

try:
    df = pd.read_excel(up)
except Exception as e:
    st.error(f"No pude leer el Excel: {e}")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Faltan columnas: {missing}")
    st.stop()

# Tipos
for c in ["puntaje google","latitud","longitud","productos disponibles","productos esperados","osa"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["_fecha_dt"] = parse_fecha_col(df["fecha"])
df = df.sort_values(["_fecha_dt","id"]).reset_index(drop=True)

# Selector de día
dias = df["_fecha_dt"].dropna().dt.date.unique().tolist()
if not dias:
    dias_serial = sorted(df["fecha"].unique().tolist())
    day_choice = st.selectbox("Elige el día (serial Excel)", dias_serial)
    df_day = df[df["fecha"] == day_choice].copy()
    st.write(f"**Día (serial):** {day_choice}")
else:
    day_choice = st.selectbox("Elige el día", dias)
    df_day = df[df["_fecha_dt"].dt.date == day_choice].copy()
    st.write(f"**Día:** {day_choice}")

st.dataframe(df_day[["id","local","distrito","latitud","longitud","productos disponibles","productos esperados","osa","estrato"]], use_container_width=True)

# ---- Ejecutar ----
if run_btn:
    # Normaliza pesos para mantener escala estable
    total_w = max(1e-9, w_osa + w_estr + w_gap)
    w_osa_n, w_estr_n, w_gap_n = w_osa/total_w, w_estr/total_w, w_gap/total_w

    priority, pnorm = compute_priority(df_day, w_osa_n, w_estr_n, w_gap_n, ESTRATO_WEIGHT_DEFAULT)
    route, dist_km, method_pretty, obj_val, points, D = solve_route(
        df_day, depot_lat, depot_lon, priority, lambda_prior, alpha_heur
    )

    if not route:
        st.warning("No se encontró una ruta (¿hay tiendas válidas?)")
        st.stop()

    tbl = build_route_table(df_day, route, priority, pnorm)
    st.subheader("Ruta sugerida (orden y prioridad)")
    st.dataframe(tbl, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Paradas", len(route))
    c2.metric("Distancia total (km, ida+vuelta)", f"{dist_km:.2f}")
    c3.metric("Método", method_pretty)
    if obj_val is not None and method_pretty.startswith("Óptimo"):
        st.caption(f"Valor objetivo (dist + λ·prioridad): {obj_val:.3f}")

    # Leyenda simple
    st.markdown("""
**Leyenda de colores:**  
- **Verde** = baja prioridad (menos urgencia)  
- **Amarillo/Naranja** = prioridad media  
- **Rojo** = alta prioridad (más urgente)  
- **Círculo más grande** = mayor prioridad  
**Números** sobre cada punto = orden de visita.
""")

    # Mapa claro
    deck = build_map(points, route, df_day, pnorm, depot_name="Depot (ArcaContinental)")
    st.pydeck_chart(deck, use_container_width=True)

    # Descargar CSV
    out_csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar ruta (CSV)", out_csv, file_name="ruta_sugerida.csv", mime="text/csv")

    st.info("Regla usada: **menor OSA**, **estrato más vulnerable (D>C>B>A)** y **mayor brecha** (esperados − disponibles) ⇒ se atienden antes. Ajusta los **pesos** y **λ/alpha** para cambiar el comportamiento.")