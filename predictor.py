import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

SECTOR_LIST = [
    "Agriculture",
    "Land-use change and forestry",
    "Waste",
    "Buildings",
    "Industry",
    "Manufacturing and construction",
    "Transport",
    "Electricity and heat",
    "Fugitive emissions",
    "Other fuel combustion",
    "Bunker fuels",
]

SECTOR_MAP_CSV = {
    "Agriculture": "agriculture",
    "Land-use change and forestry": "land_use_forestry",
    "Waste": "waste",
    "Buildings": "buildings",
    "Industry": "industry",
    "Manufacturing and construction": "manufacturing_construction",
    "Transport": "transport",
    "Electricity and heat": "electricity_heat",
    "Fugitive emissions": "fugitive_energy",
    "Other fuel combustion": "other_fuel",
    "Bunker fuels": "bunker_fuels",
}


# obtiene la mejor prediccion
def _extract_pred(res) -> Optional[np.ndarray]:
    if res is None:
        return None
    try:
        cand = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else res
        arr = np.asarray(cand, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr
    except Exception:
        return None


# genera el ruido en las predicciones
def _generate_noise(n_years, sigma=0.11, rho=0.6):
    noise = np.zeros(n_years)
    noise[0] = np.random.normal(0, sigma)
    for t in range(1, n_years):
        noise[t] = rho * noise[t - 1] + np.random.normal(0, sigma * np.sqrt(1 - rho**2))
    return noise


# predice para el modelo mas simple
def predict_simple(
    model,
    loads: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    inertia: float = 0.2,
) -> List[float]:

    denom = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
    Xf_norm = (np.asarray(loads, dtype=float) - x_min) / denom
    Xf_norm = Xf_norm.reshape(-1, 1)

    if not hasattr(model, "predict"):
        raise ValueError("Modelo inválido.")
    res = model.predict(Xf_norm)
    y_pred_norm = _extract_pred(res)
    if y_pred_norm is None:
        raise ValueError("Error inferencia.")

    y_raw = (y_pred_norm * (y_max - y_min) + y_min).flatten()  # type: ignore

    y_physics = np.zeros_like(y_raw)
    if len(y_physics) > 0:
        y_physics[0] = y_raw[0]
        for t in range(1, len(y_raw)):
            y_physics[t] = (1 - inertia) * y_raw[t] + inertia * y_physics[t - 1]

    climate_noise = _generate_noise(len(y_raw), sigma=0.11, rho=0.6)

    return [float(v) for v in (y_physics + climate_noise)]


# predice para el modelo mas complejo
def simulate_multi_sector(
    model,
    years: int,
    sector_weights: Dict[str, float],
    inertia: float = 0.2,
    start_delay: int = 5,
    transition_speed: int = 20,
) -> Tuple[List[int], List[float]]:

    try:
        df_tmp = pd.read_csv("data/temperature-anomaly.csv")
        df_ems = pd.read_csv("data/ghg-emissions-by-sector.csv")
    except FileNotFoundError:
        return [], []

    df_tmp = df_tmp[(df_tmp["Entity"] == "World") & (df_tmp["Year"] >= 1990)]
    df_tmp = df_tmp.rename(columns={"Year": "year", "Global": "temp_anomaly"})
    df_tmp = df_tmp[["year", "temp_anomaly"]][:-3]

    df_ems = df_ems[df_ems["Entity"] == "World"]

    csv_cols_map = {"Year": "year", **SECTOR_MAP_CSV}

    existing = [c for c in csv_cols_map.keys() if c in df_ems.columns]
    df_ems = df_ems[existing].rename(columns={k: csv_cols_map[k] for k in existing})

    df = pd.merge(df_tmp, df_ems, on="year", how="inner")

    internal_cols = [c for c in df_ems.columns if c != "year"]
    X_cols = []
    for col in internal_cols:
        df[f"cum_{col}"] = df[col].cumsum()
        X_cols.append(f"cum_{col}")

    x_mins = df[X_cols].min().to_numpy(dtype=float)
    x_maxs = df[X_cols].max().to_numpy(dtype=float)
    y_min, y_max = float(df["temp_anomaly"].min()), float(df["temp_anomaly"].max())

    last_idx = df.index[-1]
    year_val = df.loc[last_idx, "year"]
    current_year: int = (
        int(year_val.item()) if hasattr(year_val, "item") else int(year_val)  # type: ignore
    )
    last_cum = df.loc[last_idx, X_cols].values.astype(float)
    last_annual = df.loc[last_idx, internal_cols].values.astype(float)

    future_years = np.arange(current_year + 1, current_year + 1 + years)
    n = len(future_years)

    transition = np.zeros(n)
    start_i: int = max(0, int(start_delay))
    end_i: int = min(n, int(start_delay + transition_speed))
    if end_i > start_i:
        transition[start_i:end_i] = np.linspace(0, 1, end_i - start_i)
        transition[end_i:] = 1.0

    future_X_cum = np.zeros((n, len(internal_cols)))

    for i, col_internal in enumerate(internal_cols):
        ui_name: Optional[str] = next(
            (k for k, v in SECTOR_MAP_CSV.items() if v == col_internal), None
        )
        reduction_pct: float = sector_weights.get(ui_name or "", 0.0)
        policy_factor = 1.0 - (reduction_pct / 100.0)
        bau = last_annual[i] * (1.005 ** np.arange(n))
        current_factor = 1 - (transition * (1 - policy_factor))
        annual_curve = bau * current_factor

        future_X_cum[:, i] = last_cum[i] + np.cumsum(annual_curve)

    denom = x_maxs - x_mins
    denom[denom == 0] = 1.0
    X_input = np.clip((future_X_cum - x_mins) / denom, 0.0, 1.0)

    try:
        res = model.predict(X_input)
    except:
        res = np.array([model.predict(x.reshape(1, -1)) for x in X_input])

    y_pred_norm = _extract_pred(res)
    if y_pred_norm is None:
        return future_years.tolist(), [0.0] * n

    y_base = (y_pred_norm.flatten() * (y_max - y_min)) + y_min

    total_cum_future = np.sum(future_X_cum, axis=1)
    total_cum_history = np.sum(last_cum)
    delta_emissions = total_cum_future - total_cum_history

    tcre_per_tonne = 0.45 / 1e12
    warming_trend = delta_emissions * tcre_per_tonne

    y_raw = y_base + warming_trend

    y_final = np.zeros_like(y_raw)
    y_final[0] = y_raw[0]
    for t in range(1, n):
        y_final[t] = (1 - inertia) * y_raw[t] + inertia * y_final[t - 1]

    noise = _generate_noise(n, sigma=0.11, rho=0.6)

    return future_years.tolist(), [float(v) for v in (y_final + noise)]
