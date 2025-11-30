"""
utec_forecast_benchmark.py
---------------------------------------------------------
Tambo UTEC — Forecast 7d + Acumulación en histórico (silver)
... (resto de tu docstring) ...
"""

from __future__ import annotations
import os
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor

# ========= RUTAS (ajusta si hace falta) =========
BASE_DIR = "C:\Users\illes\Desktop\stockea"
INPUT_PATH = os.path.join(BASE_DIR, "data/1_bronze/osa_hist_Tambo_UTEC.xlsx")
OUT_FORECAST_DIR = os.path.join(BASE_DIR, "data/1_bronze/forecast")
OUT_SILVER_DIR = os.path.join(BASE_DIR, "data/2_silver")

OUT_FORECAST_XLSX = os.path.join(OUT_FORECAST_DIR, "forecast_UTEC_7d_best.xlsx")
OUT_BACKTEST_CSV = os.path.join(OUT_FORECAST_DIR, "backtest_results_UTEC.csv")
OUT_METADATA_JSON = os.path.join(OUT_FORECAST_DIR, "forecast_metadata_UTEC.json")

# Archivo acumulado en silver (histórico + predicciones)
OUT_SILVER_HISTORY = os.path.join(
    OUT_SILVER_DIR, "osa_hist_Tambo_UTEC_with_forecast.xlsx"
)

# ========= CONFIG =========
H = 7  # horizonte de pronóstico
FOLDS = 8  # nº de orígenes para walk-forward (solo si benchmark)
SEED = 42

# >>> MODO RÁPIDO: solo ExtraTrees (no corre benchmark)
SKIP_BENCHMARK = True  # True = usa ExtraTrees directo; False = corre todos los modelos
FORCE_MODEL = "ExtraTrees"  # nombre del modelo cuando SKIP_BENCHMARK=True


# ========= UTILIDADES =========
def parse_fecha(x):
    """Convierte '45,292' o 45292 (serial Excel) a datetime; si ya es fecha, la devuelve."""
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
    except Exception:
        return pd.to_datetime(s, errors="coerce")


# ==================================================================
# =================== INICIO DE LA CORRECCIÓN ======================
# ==================================================================
def load_clean_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # ESTA LÍNEA ES LA CORRECCIÓN:
    # Si tu Excel tiene columnas duplicadas (ej. "OSA" y "osa"),
    # pandas las lee como un DataFrame.
    # Esta línea fuerza a pandas a quedarse solo con la PRIMERA
    # columna de cada nombre, eliminando duplicados.
    if df.columns.duplicated().any():
        print(f"ADVERTENCIA: Columnas duplicadas detectadas en {path}. Usando la primera aparición.")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    if "fecha" not in df.columns:
        raise ValueError("No se encontró columna 'fecha' en el archivo.")
    df["fecha"] = df["fecha"].apply(parse_fecha)
    df = df.sort_values("fecha").set_index("fecha").asfreq("D")

    # numéricos base
    for c in [
        "productos esperados",
        "productos disponibles",
        "osa",
        "puntaje google",
        "latitud",
        "longitud",
    ]:
        if c in df.columns:
            # Esta línea ahora es segura y no dará error
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

    # OSA recalculada (para histórico, NO entra al modelo)
    df["osa"] = (
        df["productos disponibles"] / df["productos esperados"] * 100
    ).replace([np.inf, -np.inf], np.nan)

    # día de semana
    df["dow"] = df.index.dayofweek
    return df
# ==================================================================
# ===================== FIN DE LA CORRECCIÓN =======================
# ==================================================================


def build_supervised(df: pd.DataFrame, h: int = H):
    """
    Arma features y targets y devuelve (X_all, Y_all, X_cols, sup_df).
    Importante: OSA y sus lags NO se usan como input al modelo.
    """
    data = df.copy()

    # quitar columnas no necesarias para el modelo
    drop_cols = ["id", "local", "distrito", "puntaje google", "latitud", "longitud"]
    for c in drop_cols:
        if c in data.columns:
            data = data.drop(columns=c)

    # asegurar numéricos base (solo físicos)
    for c in ["productos disponibles", "productos esperados"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # codificación cíclica del día de semana
    data["dow_sin"] = np.sin(2 * np.pi * data["dow"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["dow"] / 7.0)

    # lags de productos disponibles (target base)
    for L in [1, 2, 3, 7, 14]:
        data[f"disp_lag{L}"] = data["productos disponibles"].shift(L)

    # lags de productos esperados
    for L in [1, 7, 14]:
        data[f"esp_lag{L}"] = data["productos esperados"].shift(L)

    # rolling stats sobre disponibles y esperados
    data["disp_roll7_mean"] = data["productos disponibles"].rolling(7).mean()
    data["disp_roll7_std"] = data["productos disponibles"].rolling(7).std()
    data["esp_roll7_mean"] = data["productos esperados"].rolling(7).mean()

    # NO se crean lags de OSA; OSA no entra al modelo

    # targets multi-step: productos disponibles futuros
    for k in range(1, h + 1):
        data[f"y_tplus{k}"] = data["productos disponibles"].shift(-k)

    sup = data.dropna().copy()
    Y_cols = [f"y_tplus{k}" for k in range(1, h + 1)]

    # Excluir del input: columnas base + OSA + dow + targets
    X_cols = [
        c
        for c in sup.columns
        if c not in ["productos disponibles", "productos esperados", "osa", "dow"]
        + Y_cols
    ]

    X_all = sup[X_cols].astype(float)
    Y_all = sup[Y_cols].astype(float)
    return X_all, Y_all, X_cols, sup


def baseline_predict(x_row: pd.Series, kind: str, h: int = H) -> np.ndarray:
    if kind == "lag1":
        return np.repeat(float(x_row["disp_lag1"]), h)
    if kind == "lag7":
        return np.repeat(float(x_row["disp_lag7"]), h)
    if kind == "pattern":
        out = []
        for k in range(1, h + 1):
            col = f"disp_lag{k}"
            out.append(float(x_row[col] if col in x_row.index else x_row["disp_lag7"]))
        return np.array(out)
    raise ValueError(kind)


# ========= DEFINIR MODELOS (solo se usa si SKIP_BENCHMARK=False) =========
def make_models():
    models = {}

    models["RF"] = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )

    models["HGB"] = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_depth=None,
            max_iter=500,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=SEED,
        )
    )

    try:
        from xgboost import XGBRegressor

        models["XGB"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=700,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=4,
                tree_method="hist",
            )
        )
    except Exception as e:
        print("XGBoost no disponible:", e)

    try:
        from lightgbm import LGBMRegressor

        models["LGBM"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=900,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        )
        models["LGBM_tuned"] = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=127,
                min_child_samples=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=0.0,
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    except Exception as e:
        print("LightGBM no disponible:", e)

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                loss_function="RMSE",
                random_seed=SEED,
                verbose=False,
            )
        )
    except Exception as e:
        print("CatBoost no disponible:", e)

    return models


# ========= BACKTEST (solo si SKIP_BENCHMARK=False) =========
def backtest_models(
    X_all: pd.DataFrame,
    Y_all: pd.DataFrame,
    models: dict,
    h: int = H,
    folds: int = FOLDS,
):
    n = len(X_all)
    if n < folds * h + 1:
        raise ValueError(
            f"No hay suficientes filas para {folds} ventanas de {h} días. Filas={n}"
        )

    origins = [n - folds * h + i * h for i in range(folds)]
    names = ["baseline_lag1", "baseline_lag7", "baseline_pattern", *models.keys()]
    results_sse = {name: np.zeros(h) for name in names}
    count = 0

    for pos in origins:
        X_tr, Y_tr = X_all.iloc[:pos], Y_all.iloc[:pos]
        X_o = X_all.iloc[[pos]]
        y_true_vec = Y_all.iloc[pos].values.astype(float)  # (h,)

        # baselines
        for bname, kind in [
            ("baseline_lag1", "lag1"),
            ("baseline_lag7", "lag7"),
            ("baseline_pattern", "pattern"),
        ]:
            y_pred_vec = baseline_predict(X_o.iloc[0], kind, h=h)
            results_sse[bname] += (y_true_vec - y_pred_vec) ** 2

        # modelos
        for name, est in models.items():
            est.fit(X_tr, Y_tr)
            y_pred_vec = est.predict(X_o)[0].astype(float)
            results_sse[name] += (y_true_vec - y_pred_vec) ** 2

        count += 1

    rows = []
    rmse_by_model = {}
    for name, sse in results_sse.items():
        rmse_h = np.sqrt(sse / count)
        rmse_by_model[name] = rmse_h
        rows.append((name, rmse_h[0], rmse_h[-1], float(rmse_h.mean())))

    res_bt = pd.DataFrame(
        rows, columns=["modelo", "RMSE@1", "RMSE@7", "RMSE_mean(1..7)"]
    ).sort_values("RMSE_mean(1..7)")
    best_name = res_bt.iloc[0]["modelo"]
    return res_bt, best_name, rmse_by_model


# ========= FORECAST 7D (enteros + clip + OSA) =========
def forecast_next_7d(df: pd.DataFrame, X_cols: list[str], estimator) -> pd.DataFrame:
    """
    Genera forecast 7 días usando el mismo esquema de features que build_supervised.
    Importante: no usa OSA ni lags de OSA como input del modelo.
    OSA_pred se calcula SOLO a partir de prod_disp_pred y prod_esp_ref.
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

    # última fecha conocida
    t0 = data.index.max()
    X_last = data.loc[[t0], X_cols].astype(float)
    yhat = estimator.predict(X_last)[0].astype(float)  # shape (H,)

    future_dates = pd.date_range(t0 + pd.Timedelta(days=1), periods=H, freq="D")

    # productos esperados de referencia para futuro (semana previa o último valor)
    exp_future = []
    for d in future_dates:
        ref = d - pd.Timedelta(days=7)
        if ref in df.index and not pd.isna(df.loc[ref, "productos esperados"]):
            exp_future.append(int(df.loc[ref, "productos esperados"]))
        else:
            exp_future.append(int(df["productos esperados"].iloc[-1]))
    exp_future = np.array(exp_future, dtype=int)

    # predicción de disponibles, redondeo y clip [0, prod_esp_ref]
    yhat_int = np.rint(yhat).astype(int)
    yhat_int = np.clip(yhat_int, 0, exp_future)

    # OSA calculada SOLO a partir de predicción y esperados
    osa_pred = np.round((yhat_int / np.maximum(exp_future, 1)) * 100.0, 2)

    out = pd.DataFrame(
        {
            "fecha": future_dates,
            "prod_esp_ref": exp_future,
            "prod_disp_pred": yhat_int,
            "OSA_pred": osa_pred,
        }
    )
    return out


# ========= IMPORTANCIAS =========
def feature_importances_safely(model, X_cols: list[str]) -> pd.DataFrame | None:
    try:
        if hasattr(model, "estimators_"):  # ExtraTrees/RandomForest
            fi = np.mean(
                [est.feature_importances_ for est in model.estimators_], axis=0
            )
            return (
                pd.DataFrame({"feature": X_cols, "importance": fi})
                .sort_values("importance", ascending=False)
            )
        if hasattr(model, "estimators"):
            fis = []
            for (
                est
            ) in (
                model.estimators_
                if hasattr(model, "estimators_")
                else model.estimators
            ):
                if hasattr(est, "feature_importances_"):
                    fis.append(est.feature_importances_)
            if fis:
                fi = np.mean(np.vstack(fis), axis=0)
                return (
                    pd.DataFrame({"feature": X_cols, "importance": fi})
                    .sort_values("importance", ascending=False)
                )
    except Exception:
        pass
    return None


# ========= APÉNDICE / ACUMULACIÓN EN SILVER =========
def _next_ids(last_id: str, n: int) -> list[str] | None:
    """Genera n IDs consecutivos si el último sigue patrón Prefix + dígitos."""
    if not isinstance(last_id, str):
        return None
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", last_id.strip())
    if not m:
        return None
    prefix, num = m.group(1), int(m.group(2))
    width = len(m.group(2))
    return [f"{prefix}{str(num + i).zfill(width)}" for i in range(1, n + 1)]


def append_forecast_to_history(
    base_df: pd.DataFrame, forecast_df: pd.DataFrame, silver_path: str
) -> pd.DataFrame:
    """
    Acumula predicciones al histórico en un archivo 'silver'.
    - Si el archivo silver ya existe: se lee y se actualiza sin duplicar fechas.
      Se eliminan filas con fecha >= min(forecast_df.fecha) y se añaden las nuevas predicciones.
    - Si no existe: se usa el histórico base_df como punto de partida.
    Devuelve el DataFrame final y lo guarda en 'silver_path'.
    """
    # 1) Cargar silver existente si hay
    if os.path.exists(silver_path):
        silver_df = pd.read_excel(silver_path)
        silver_df.columns = [c.strip().lower() for c in silver_df.columns]
        
        # Manejar columnas duplicadas por si acaso
        if silver_df.columns.duplicated().any():
            silver_df = silver_df.loc[:, ~silver_df.columns.duplicated(keep='first')]

        silver_df["fecha"] = silver_df["fecha"].apply(parse_fecha)
        silver_df = silver_df.sort_values("fecha").reset_index(drop=True)
    else:
        silver_df = base_df.reset_index().copy()  # incluye 'fecha'
        silver_df = silver_df.reset_index(drop=True)

    # 2) Preparar constantes (del último registro conocido en silver)
    const_vals = {}
    for c in ["local", "distrito", "latitud", "longitud", "puntaje google"]:
        if c in silver_df.columns and silver_df[c].notna().any():
            const_vals[c] = silver_df[c].dropna().iloc[-1]
        elif c in base_df.columns and base_df[c].notna().any():
            const_vals[c] = base_df[c].dropna().iloc[-1]
        else:
            const_vals[c] = np.nan

    # 3) Calcular IDs futuros continuando los existentes si aplica
    new_ids = None
    if "id" in silver_df.columns and silver_df["id"].notna().any():
        last_id_val = silver_df["id"].dropna().iloc[-1]
        new_ids = _next_ids(str(last_id_val), len(forecast_df))

    # 4) Convertir forecast_df a esquema de columnas del histórico
    rows = []
    for _, row in forecast_df.iterrows():
        r = {
            "fecha": row["fecha"],
            "productos disponibles": int(row["prod_disp_pred"]),
            "productos esperados": int(row["prod_esp_ref"]),
            # OSA se guarda a partir de la predicción
            "osa": float(row["OSA_pred"]),
        }
        for k, v in const_vals.items():
            r[k] = v
        if new_ids is not None:
            r["id"] = new_ids[len(rows)]
        rows.append(r)
    append_df = pd.DataFrame(rows)

    # 5) Evitar duplicados: remover en silver fechas >= primera fecha del forecast
    first_fc_date = forecast_df["fecha"].min()
    silver_df = silver_df[silver_df["fecha"] < first_fc_date].copy()

    # 6) Concatenar y ordenar
    out_all = pd.concat([silver_df, append_df], ignore_index=True, sort=False)
    out_all = out_all.sort_values("fecha").reset_index(drop=True)

    # 7) Reordenar columnas al orden deseado si existen
    preferred = [
        "fecha",
        "id",
        "local",
        "distrito",
        "puntaje google",
        "latitud",
        "longitud",
        "productos disponibles",
        "productos esperados",
        "osa",
    ]
    others = [c for c in out_all.columns if c not in preferred]
    cols = [c for c in preferred if c in out_all.columns] + others
    out_all = out_all[cols]

    # 8) Guardar
    os.makedirs(os.path.dirname(silver_path), exist_ok=True)
    out_all.to_excel(silver_path, index=False)
    return out_all


# ========= FUNCIÓN PÚBLICA PARA TU APP =========
def run_forecast():
    """
    Ejecuta todo el flujo de forecast:
    - lee histórico UTEC
    - entrena modelo
    - genera forecast 7d
    - actualiza silver
    - guarda archivos en disco

    Devuelve:
        forecast_df, hist_plus_fc, meta_dict
    """
    os.makedirs(OUT_FORECAST_DIR, exist_ok=True)
    os.makedirs(OUT_SILVER_DIR, exist_ok=True)

    print("Cargando y limpiando datos…")
    df = load_clean_df(INPUT_PATH)
    print(f"Rango: {df.index.min().date()} → {df.index.max().date()}  ({len(df)} días)")

    print("Construyendo dataset supervisado…")
    X_all, Y_all, X_cols, _ = build_supervised(df, h=H)
    print("Shapes:", X_all.shape, Y_all.shape)

    # --- Camino A: rápido (solo ExtraTrees)  /  Camino B: benchmark completo ---
    res_bt = None
    if SKIP_BENCHMARK:
        print("⚡ Modo rápido: ExtraTrees directo.")
        best_name = FORCE_MODEL
        best_est = ExtraTreesRegressor(
            n_estimators=800,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )
    else:
        print("Creando catálogo de modelos…")
        models = make_models()
        print("Modelos:", list(models.keys()))
        print(f"Backtesting walk-forward: FOLDS={FOLDS}, H={H}…")
        res_bt, best_name, _ = backtest_models(
            X_all, Y_all, models, h=H, folds=FOLDS
        )
        print("\n=== Resultados backtest (RMSE) ===")
        print(res_bt.round(3).to_string(index=False))
        print("\nMejor modelo en backtest:", best_name)
        if best_name.startswith("baseline_"):
            raise RuntimeError("El mejor fue un baseline. Revisa features/datos.")
        best_est = models[best_name]
        res_bt.round(6).to_csv(OUT_BACKTEST_CSV, index=False)

    print("\nEntrenando modelo seleccionado en todo el histórico…")
    best_est.fit(X_all, Y_all)

    print("Generando forecast 7 días…")
    forecast_df = forecast_next_7d(df, X_cols, best_est)
    assert (
        len(forecast_df) == H
    ), f"Se esperaban {H} filas de forecast y se obtuvieron {len(forecast_df)}."

    # (1) Guardar SOLO predicciones en BRONZE/forecast
    forecast_df.to_excel(OUT_FORECAST_XLSX, index=False)
    print("✅ Guardado SOLO predicciones en:", OUT_FORECAST_XLSX)

    # (2) Acumular predicciones en SILVER (histórico + forecast)
    hist_plus_fc = append_forecast_to_history(df, forecast_df, OUT_SILVER_HISTORY)
    print("✅ Histórico ACUMULADO actualizado en:", OUT_SILVER_HISTORY)

    # Importancias (si aplica)
    fi_df = feature_importances_safely(best_est, list(X_all.columns))
    top_features = None
    if fi_df is not None:
        top_features = fi_df.head(12).to_dict(orient="records")
        print("\nTop features (aprox):")
        print(fi_df.head(12).round(4).to_string(index=False))

    meta = {
        "best_model": best_name,
        "horizon": H,
        "folds": FOLDS if not SKIP_BENCHMARK else None,
        "input_path": INPUT_PATH,
        "output_forecast": OUT_FORECAST_XLSX,
        "output_silver_history": OUT_SILVER_HISTORY,
        "output_backtest_csv": (
            OUT_BACKTEST_CSV if (not SKIP_BENCHMARK and res_bt is not None) else None
        ),
        "generated_rows": int(forecast_df.shape[0]),
        "top_features": top_features,
    }
    with open(OUT_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("✅ Guardado metadatos en:", OUT_METADATA_JSON)

    print("\n=== RESUMEN ===")
    print("Modelo final:", best_name, "(modo rápido)" if SKIP_BENCHMARK else "(benchmark)")
    print("Forecast 7d (t+1..t+7):")
    print(forecast_df.to_string(index=False))

    return forecast_df, hist_plus_fc, meta


# ========= MAIN CLI =========
if __name__ == "__main__":
    run_forecast()