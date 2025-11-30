# generar_hist_osa_sintetica.py
# ---------------------------------------------------------
# Genera historial OSA diario (01-01-2024 .. hoy) para:
#   - Tambo Cardenas
#   - Tambo UTEC
# usando como semillas una fila de
# /Users/johar/Desktop/Desarrollo/proyecto_dpd/data/bronze/osa_resultados.xlsx
#
# Salida: un Excel por tienda en la carpeta "bronze":
#   osa_hist_Tambo_Cardenas.xlsx
#   osa_hist_Tambo_UTEC.xlsx
# ---------------------------------------------------------

from __future__ import annotations
import os
from datetime import date
import numpy as np
import pandas as pd

# ---------- Config ----------
INPUT_XLSX = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/bronze/osa_resultados.xlsx"
OUTPUT_DIR = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/1_bronze"

TIENDAS_OBJETIVO = ["Tambo UTEC"]
FECHA_INICIO = date(2024, 1, 1)    # lunes
FECHA_FIN = date.today()           # hoy
SEED = 123
np.random.seed(SEED)

# Estacionalidad semanal (Mon..Sun)
DOW_MULT = np.array([1.00, 1.03, 1.05, 1.02, 1.10, 0.90, 0.85])

# (Opcional) Feriados PE 2024-2025 para variar un poco OSA
FERIADOS = set(pd.to_datetime([
    "2024-01-01","2024-03-28","2024-03-29","2024-05-01","2024-06-29",
    "2024-07-28","2024-07-29","2024-08-30","2024-10-08","2024-11-01",
    "2024-12-08","2024-12-25",
    "2025-01-01","2025-04-17","2025-04-18","2025-05-01","2025-06-29",
    "2025-07-28","2025-07-29","2025-08-30","2025-10-08","2025-11-01",
    "2025-12-08","2025-12-25",
]).date)

# ---------- Utilidades ----------
def split_id_prefix_num(s: str):
    """Devuelve (prefijo_letras, numero, ancho_digitos) a partir de p.ej. 'TCL0001'."""
    s = str(s)
    i = 0
    while i < len(s) and not s[i].isdigit():
        i += 1
    pref = s[:i]
    digits = s[i:] or "1"
    width = len(digits)
    num = int(digits)
    return pref, num, width

def zero_pad(n: int, width: int) -> str:
    return str(n).zfill(width)

def cargar_semilla(path: str, local: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo base: {path}")
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    req = ["id","local","distrito","puntaje google","latitud","longitud",
           "productos disponibles","productos esperados","osa"]
    faltan = [c for c in req if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el Excel base: {faltan}")

    row = df[df["local"].str.strip().str.lower() == local.lower()].head(1)
    if row.empty:
        raise ValueError(f"No se encontró una fila semilla para el local: {local}")

    r = row.iloc[0].to_dict()
    # normaliza tipos
    return {
        "id": str(r["id"]),
        "local": str(r["local"]),
        "distrito": str(r["distrito"]),
        "puntaje google": float(r["puntaje google"]),
        "latitud": float(r["latitud"]),
        "longitud": float(r["longitud"]),
        "base_disponibles": int(r["productos disponibles"]),
        "base_esperados": int(r["productos esperados"]),
        "base_osa": float(r["osa"]),
    }

def generar_historial(local: str, semilla: dict) -> pd.DataFrame:
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    n = len(fechas)
    dow = fechas.weekday.values  # 0..6

    # Esperados con estacionalidad semanal + ruido
    base_esp = semilla["base_esperados"]
    esperados = base_esp * DOW_MULT[dow] * np.random.normal(1.0, 0.05, n)
    esperados = np.clip(np.round(esperados), 1, None).astype(int)

    # OSA con pequeñas variaciones alrededor de la base,
    # bajando un poco en feriados, subiendo levemente en viernes
    base_osa = semilla["base_osa"]
    is_holiday = np.array([int(f.date() in FERIADOS) for f in fechas])
    is_friday = (dow == 4).astype(int)

    osa = (
        base_osa
        - 4.0 * is_holiday
        + 2.0 * is_friday
        + np.random.normal(0, 2.5, n)   # ruido (±2.5pp aprox)
    )
    osa = np.clip(osa, 60.0, 100.0)

    # Disponibles a partir de OSA y esperados
    disponibles = np.minimum(esperados, np.round(esperados * (osa / 100.0)).astype(int))

    # IDs incrementales por día (día 1 => ...0001)
    pref, _, width = split_id_prefix_num(semilla["id"])
    pad = width if width > 0 else 4
    ids = [f"{pref}{zero_pad(i, pad)}" for i in range(1, n + 1)]

    # Construcción del DataFrame final
    df = pd.DataFrame({
        "fecha": fechas.date,
        "id": ids,
        "local": local,
        "distrito": semilla["distrito"],
        "puntaje google": round(semilla["puntaje google"], 2),
        "latitud": semilla["latitud"],
        "longitud": semilla["longitud"],
        "productos disponibles": disponibles,
        "productos esperados": esperados,
    })
    df["OSA"] = (df["productos disponibles"] / df["productos esperados"] * 100).round(2)

    cols = ["fecha","id","local","distrito","puntaje google","latitud","longitud",
            "productos disponibles","productos esperados","OSA"]
    return df[cols]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for local in TIENDAS_OBJETIVO:
        sem = cargar_semilla(INPUT_XLSX, local)
        df_hist = generar_historial(local, sem)

        # Archivo por tienda
        safe_local = "".join(ch if ch.isalnum() else "_" for ch in local)
        out_path = os.path.join(OUTPUT_DIR, f"osa_hist_{safe_local}.xlsx")
        df_hist.to_excel(out_path, index=False)
        print(f"✅ Guardado: {out_path}  ({len(df_hist)} filas)")

if __name__ == "__main__":
    main()