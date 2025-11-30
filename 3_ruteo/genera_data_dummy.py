# Fuente (INEI, planos estratificados / referencia NSE):
# https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1744/libro.pdf

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# ------------------ Rutas ------------------
SILVER_PATH = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/2_silver/osa_hist_Tambo_UTEC_with_forecast.xlsx"
GOLD_DIR    = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/3_gold"
GOLD_PATH   = os.path.join(GOLD_DIR, "gold_tiendas_7d.xlsx")

# ------------------ Utilidades ------------------
def parse_fecha(x):
    """Convierte '45,292' o 45292 (serial Excel) a datetime; si ya es fecha, la devuelve."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x)
    s = str(x).strip().replace(",", "")
    try:
        n = int(float(s))
        # rango razonable de seriales (1990–2100)
        if 30000 <= n <= 60000:
            return pd.to_datetime(n, unit="D", origin="1899-12-30")
        return pd.to_datetime(s, errors="coerce")
    except:
        return pd.to_datetime(s, errors="coerce")

def std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def ensure_cols(df: pd.DataFrame, required: list[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el silver: {missing}")

# ------------------ Carga UTEC (últimos 7 días) ------------------
def load_utec_last7(silver_path: str) -> pd.DataFrame:
    if not os.path.isfile(silver_path):
        raise FileNotFoundError(f"No se encontró el archivo silver en: {silver_path}")

    df = pd.read_excel(silver_path)
    df = std_cols(df)
    if "fecha" not in df.columns:
        raise ValueError("El archivo silver no tiene columna 'fecha'.")
    df["fecha"] = df["fecha"].apply(parse_fecha)

    # Requisitos mínimos
    base_cols = [
        "fecha","id","local","distrito","puntaje google","latitud","longitud",
        "productos disponibles","productos esperados","osa"
    ]
    ensure_cols(df, [c for c in base_cols if c in df.columns or True])  # tolerante si faltan id/local/etc., pero avisará luego si no están

    # Filtro a UTEC (por id o por nombre del local)
    is_utec = (df.get("id","").astype(str).str.upper() == "TUB0001") | \
              (df.get("local","").astype(str).str.contains("UTEC", case=False, na=False))

    utec_df = df[is_utec].copy()
    if utec_df.empty:
        # Si no identifica por id/local, asumimos que TODO el silver es UTEC (según tu pipeline previo)
        utec_df = df.copy()

    utec_df = utec_df.sort_values("fecha")
    last7 = utec_df.tail(7).copy()

    # Validar que tengamos las columnas clave
    need = ["fecha","productos disponibles","productos esperados"]
    ensure_cols(last7, need)

    # Completar metadatos de UTEC si faltan (tomamos el último valor no nulo)
    for c in ["id","local","distrito","puntaje google","latitud","longitud","osa"]:
        if c not in last7.columns:
            last7[c] = np.nan
        if last7[c].isna().all() and c in utec_df.columns and utec_df[c].notna().any():
            last7[c] = utec_df[c].dropna().iloc[-1]

    # Si aún falta id/local, definimos por defecto
    if last7["id"].isna().all():
        last7["id"] = "TUB0001"
    if last7["local"].isna().all():
        last7["local"] = "Tambo UTEC"
    if last7["distrito"].isna().all():
        last7["distrito"] = "Barranco"

    # Puntaje/coords/OSA: mantener lo que venga o completar con NaN si no hay
    return last7[[
        "fecha","id","local","distrito","puntaje google","latitud","longitud",
        "productos disponibles","productos esperados","osa"
    ]].reset_index(drop=True)

# ------------------ Simulación para otras tiendas ------------------
STORE_META = [
    {"id":"TCL0001","local":"Tambo Cardenas","distrito":"Lince","puntaje google":3.8,"latitud":-12.07700095532361,"longitud":-77.03451000442836,"estrato":"B"},
    {"id":"TCLV0001","local":"Tambo Canada","distrito":"La Victoria","puntaje google":3.7,"latitud":-12.082461678662392,"longitud":-77.01266436414203,"estrato":"C"},
    {"id":"TAM0001","local":"Tambo Angamos","distrito":"Miraflores","puntaje google":4.0,"latitud":-12.113605612152355,"longitud":-77.03035891795264,"estrato":"A"},
    {"id":"TMEA0001","local":"Tambo Mariategui","distrito":"El Agustino","puntaje google":2.3,"latitud":-12.03002783443611,"longitud":-76.99889893732406,"estrato":"D"},
]

# Factores deterministas por estrato para simular OSA (puedes ajustar)
# La idea es que A > B > C > D en cumplimiento: disponibles ≈ factor * esperados
FACTOR_BY_ESTRATO = {"A":0.78, "B":0.74, "C":0.68, "D":0.62}

def simulate_for_stores(utec_last7: pd.DataFrame) -> pd.DataFrame:
    # Fechas base: las mismas 7 fechas de UTEC
    dates = list(utec_last7["fecha"])
    # Tomamos como referencia los "productos esperados" de UTEC
    exp_ref = list(utec_last7["productos esperados"].astype(float))

    rows = []
    rng = np.random.default_rng(42)  # reproducible
    for meta in STORE_META:
        estrato = meta["estrato"]
        factor  = FACTOR_BY_ESTRATO.get(estrato, 0.70)

        # Pequeña variación determinista para no calcarlas (±2 con tope)
        jitter = rng.integers(-2, 3, size=len(dates))  # -2..+2

        for i, d in enumerate(dates):
            esp = int(max(1, round(exp_ref[i] + jitter[i])))
            disp = int(min(esp, max(0, round(esp * factor))))
            osa  = round((disp / max(esp,1)) * 100.0, 2)

            rows.append({
                "fecha": d,
                "id": meta["id"],
                "local": meta["local"],
                "distrito": meta["distrito"],
                "puntaje google": meta["puntaje google"],
                "latitud": meta["latitud"],
                "longitud": meta["longitud"],
                "productos disponibles": disp,
                "productos esperados": esp,
                "osa": osa,
                "estrato": estrato
            })

    sim_df = pd.DataFrame(rows)
    return sim_df

# ------------------ Main ------------------
def main():
    os.makedirs(GOLD_DIR, exist_ok=True)

    # 1) UTEC last 7 (reales + pred)
    utec7 = load_utec_last7(SILVER_PATH)

    # Añadir estrato a UTEC (B, según tu instrucción)
    utec7["estrato"] = "B"

    # 2) Simular otras 4 tiendas (7 registros c/u) usando mismas fechas
    sim_df = simulate_for_stores(utec7)

    # 3) Unir (orden uniforme de columnas)
    cols = ["fecha","id","local","distrito","puntaje google","latitud","longitud",
            "productos disponibles","productos esperados","osa","estrato"]

    # Asegurar tipos
    for c in ["productos disponibles","productos esperados"]:
        utec7[c] = pd.to_numeric(utec7[c], errors="coerce").fillna(0).astype(int)
    utec7["osa"] = pd.to_numeric(utec7["osa"], errors="coerce")

    # Concatenar: primero UTEC, luego simuladas
    out = pd.concat([utec7[cols], sim_df[cols]], ignore_index=True)
    out = out.sort_values(["fecha","id"]).reset_index(drop=True)

    # 4) Guardar Excel en gold
    out.to_excel(GOLD_PATH, index=False)
    print("✅ Archivo generado:", GOLD_PATH)
    print("Filas totales:", len(out))
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()