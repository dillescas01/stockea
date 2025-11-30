# generar_hist_osa_sintetica_clean.py
# ---------------------------------------------------------
# Genera historial OSA diario (2024-01-01 .. hoy) para tiendas
# usando solo STORE_META (sin depender de Excels viejos).
#
# Salida (un solo archivo limpio):
#   /Users/johar/Desktop/Desarrollo/proyecto_dpd/data/1_bronze/osa_hist_Tambo_UTEC.xlsx
#
# Columnas alineadas con la app:
#   fecha, id, local, distrito, puntaje google,
#   latitud, longitud, productos disponibles,
#   productos esperados, osa, estrato
# ---------------------------------------------------------

from __future__ import annotations
import os
from datetime import date
import numpy as np
import pandas as pd
from stores_meta import STORE_META  # usa tu diccionario

# ---------- Config ----------
OUTPUT_DIR = "/Users/johar/Desktop/Desarrollo/proyecto_dpd/data/1_bronze"
# usa los IDs que quieras generar
TIENDAS_OBJETIVO_IDS = ["TUB0001"]   # puedes agregar "TCL0001", etc.
FECHA_INICIO = date(2024, 1, 1)
FECHA_FIN = date.today()
SEED = 123
np.random.seed(SEED)

# Estacionalidad semanal (Mon..Sun)
DOW_MULT = np.array([1.00, 1.03, 1.05, 1.02, 1.10, 0.90, 0.85])

# Feriados para Perú 2024-2025 (para variar OSA un poco)
FERIADOS = set(pd.to_datetime([
    "2024-01-01","2024-03-28","2024-03-29","2024-05-01","2024-06-29",
    "2024-07-28","2024-07-29","2024-08-30","2024-10-08","2024-11-01",
    "2024-12-08","2024-12-25",
    "2025-01-01","2025-04-17","2025-04-18","2025-05-01","2025-06-29",
    "2025-07-28","2025-07-29","2025-08-30","2025-10-08","2025-11-01",
    "2025-12-08","2025-12-25",
]).date)

def split_id_prefix_num(s: str):
    """Devuelve (prefijo_letras, num, width) a partir de p.ej. 'TUB0001'."""
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

def generar_historial_para_tienda(store_id: str) -> pd.DataFrame:
    meta = STORE_META[store_id]
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    n = len(fechas)
    dow = fechas.weekday.values  # 0..6

    # Esperados con estacionalidad semanal + ruido,
    # centrados en "esperados_base" del meta
    base_esp = meta["esperados_base"]
    esperados = base_esp * DOW_MULT[dow] * np.random.normal(1.0, 0.05, n)
    esperados = np.clip(np.round(esperados), 1, None).astype(int)

    # OSA simulada alrededor de 75–90% dependiendo del estrato
    estrato = str(meta["estrato"]).upper()
    if estrato == "A":
        base_osa = 88.0
    elif estrato == "B":
        base_osa = 84.0
    elif estrato == "C":
        base_osa = 80.0
    else:  # D
        base_osa = 76.0

    is_holiday = np.array([int(f.date() in FERIADOS) for f in fechas])
    is_friday = (dow == 4).astype(int)

    osa = (
        base_osa
        - 4.0 * is_holiday
        + 2.0 * is_friday
        + np.random.normal(0, 2.5, n)
    )
    osa = np.clip(osa, 60.0, 99.0)

    # Disponibles = redondeo de esperados * OSA
    disponibles = np.minimum(
        esperados,
        np.round(esperados * (osa / 100.0)).astype(int)
    )

    # IDs diarios (TUB0001, TUB0002, ...)
    pref, _, width = split_id_prefix_num(meta["id"])
    pad = width if width > 0 else 4
    ids = [f"{pref}{zero_pad(i, pad)}" for i in range(1, n + 1)]

    df = pd.DataFrame({
        "fecha": fechas,  # datetime directo
        "id": ids,
        "local": meta["local"],
        "distrito": meta["distrito"],
        "puntaje google": float(meta["puntaje_google"]),
        "latitud": meta["latitud"],
        "longitud": meta["longitud"],
        "productos disponibles": disponibles,
        "productos esperados": esperados,
        "osa": osa.round(2),
        "estrato": estrato,
    })

    cols = [
        "fecha","id","local","distrito","puntaje google","latitud","longitud",
        "productos disponibles","productos esperados","osa","estrato"
    ]
    return df[cols]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Por ahora solo generamos una tienda; puedes concatenar varias si quieres
    hist_frames = []
    for store_id in TIENDAS_OBJETIVO_IDS:
        if store_id not in STORE_META:
            raise ValueError(f"No existe store_id={store_id} en STORE_META.")
        df_hist = generar_historial_para_tienda(store_id)
        hist_frames.append(df_hist)

    df_all = pd.concat(hist_frames, ignore_index=True)

    # Guardamos TODO en un solo archivo, que es justo el que usa la app
    out_path = os.path.join(OUTPUT_DIR, "osa_hist_Tambo_UTEC.xlsx")
    df_all.to_excel(out_path, index=False)
    print(f"✅ Guardado histórico limpio en: {out_path} ({len(df_all)} filas)")

if __name__ == "__main__":
    main()