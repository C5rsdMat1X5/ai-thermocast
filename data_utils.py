import pandas as pd
import numpy as np


# obtiene la info de los datasets
def get_data(mode: str = "simple"):
    if mode == "simple":
        df_tmp = pd.read_csv("data/temperature-anomaly.csv")
        df_tmp = df_tmp[df_tmp["Entity"] == "World"]
        df_tmp = df_tmp[["Year", "Global"]][:-1]

        df_ems = pd.read_csv("data/annual-co2-emissions-per-country.csv")
        df_ems = df_ems[df_ems["Entity"] == "World"]
        df_ems = df_ems[df_ems["Year"] >= 1850]
        df_ems = df_ems[["Year", "Emissions"]]

        df_tmp = df_tmp.rename(columns={"Year": "year", "Global": "temp_anomaly"})
        df_ems = df_ems.rename(columns={"Year": "year", "Emissions": "co2_emissions"})
        df = pd.merge(df_tmp, df_ems, on="year", how="inner")

        df["cum_emissions"] = df["co2_emissions"].cumsum()
        X = df[["cum_emissions"]]
        y = df[["temp_anomaly"]]

        x_min = float(X.min().values[0])
        x_max = float(X.max().values[0])

        y_min = float(y.min().values[0])
        y_max = float(y.max().values[0])

        denom = x_max - x_min if (x_max - x_min) != 0 else 1.0
        Xn = (X - x_min) / denom
        yn = (y - y_min) / (y_max - y_min if (y_max - y_min) != 0 else 1.0)

        X_np = Xn.to_numpy(dtype=float)
        y_np = yn.to_numpy(dtype=float)

        last_year = int(df["year"].iloc[-1])
        last_added_emissions = float(df["cum_emissions"].iloc[-1])
        last_annual = float(df["co2_emissions"].iloc[-1])

        meta = {
            "last_year": last_year,
            "last_added_emissions": last_added_emissions,
            "last_annual": last_annual,
        }

        return X_np, y_np, (x_min, x_max), (y_min, y_max), meta

    df_tmp = pd.read_csv("data/temperature-anomaly.csv")
    df_tmp = df_tmp[df_tmp["Entity"] == "World"]
    df_tmp = df_tmp[df_tmp["Year"] >= 1990]
    df_tmp = df_tmp[["Year", "Global"]][:-3]
    df_tmp = df_tmp.rename(columns={"Year": "year", "Global": "temp_anomaly"})
    df_ems = pd.read_csv("data/ghg-emissions-by-sector.csv")
    df_ems = df_ems[df_ems["Entity"] == "World"]
    df_ems = df_ems[
        [
            "Year",
            "Agriculture",
            "land-use change and forestry",
            "waste",
            "buildings",
            "industry",
            "manufacturing and construction",
            "transport",
            "electricity and heat",
            "Fugitive emissions from energy production",
            "other fuel combustion",
            "bunker fuels",
        ]
    ]
    df_ems = df_ems.rename(
        columns={
            "Year": "year",
            "Agriculture": "agriculture",
            "land-use change and forestry": "land_use_forestry",
            "waste": "waste",
            "buildings": "buildings",
            "industry": "industry",
            "manufacturing and construction": "manufacturing_construction",
            "transport": "transport",
            "electricity and heat": "electricity_heat",
            "Fugitive emissions from energy production": "fugitive_energy",
            "other fuel combustion": "other_fuel",
            "bunker fuels": "bunker_fuels",
        }
    )

    df = pd.merge(df_tmp, df_ems, on="year", how="inner")
    SECTOR_COLS = [
        "agriculture",
        "land_use_forestry",
        "waste",
        "industry",
        "buildings",
        "manufacturing_construction",
        "transport",
        "electricity_heat",
        "fugitive_energy",
        "other_fuel",
        "bunker_fuels",
    ]

    for col in SECTOR_COLS:
        df[f"cum_{col}"] = df[col].cumsum()

    X = df[[f"cum_{c}" for c in SECTOR_COLS]]
    y = df[["temp_anomaly"]]

    x_min_arr = X.min().to_numpy(dtype=float)
    x_max_arr = X.max().to_numpy(dtype=float)

    y_min_val = float(y.min().values[0])
    y_max_val = float(y.max().values[0])

    yn = (y - y_min_val) / (
        y_max_val - y_min_val if y_max_val - y_min_val != 0 else 1.0
    )

    y_min = y_min_val
    y_max = y_max_val

    X_np = (X - x_min_arr) / (x_max_arr - x_min_arr + 1e-12)
    X_np = np.clip(X_np.to_numpy(dtype=float), 0.0, 1.0)
    y_np = yn.to_numpy(dtype=float)

    meta = {"last_year": None, "last_added_emissions": None, "last_annual": None}
    return X_np, y_np, (x_min_arr, x_max_arr), (y_min, y_max), meta
