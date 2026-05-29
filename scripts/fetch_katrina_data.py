"""
SciVis 2026 — Task 2: Acoplamiento Atmosfera-Oceano
Extractor de datos reales para el Huracán Katrina (Agosto 2005)
y todo el año 2005 sobre el Atlántico Norte.

FUENTES (NASA/NOAA, todos satelitales o reanálisis observacional):
  - SST mensual:      NOAA OISST v2  (Reynolds OI 1°x1°)
  - SST climatología: NOAA OISST v2 LTM 1991-2020
  - SST diario:       NOAA OISST v2 high-res (0.25°) para track diario de Katrina
  - Viento 10m:       NCEP/NCAR Reanalysis 1 — uwnd, vwnd mensual (2.5°)
  - T aire 2m:        NCEP/NCAR Reanalysis 1 — air.mon.mean
  - Presión sup.:     NCEP/NCAR Reanalysis 1 — pres.mon.mean

NOTA SOBRE DYAMOND: la simulación DYAMOND-v2 acoplada de NASA cubre el
periodo 2020-01-19 → 2020-03-01 (DYAMOND-2 Winter). NO incluye 2005.
Para visualizar el caso real Katrina usamos los productos de reanálisis
de los cuales DYAMOND deriva sus condiciones de frontera (ERA5/MERRA-2
equivalentes), lo cual SÍ permite mostrar datos reales del Atlántico
Norte en agosto 2005.

Salida: data/dyamond_task2.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dyamond_task2.json"

# ── Región: Atlántico Norte (igual al index.html) ──
REGION = {"lon0": -80.0, "lon1": 20.0, "lat0": 10.0, "lat1": 65.0}
NX, NY = 60, 44          # mismo grid que el dashboard
WIND_NX, WIND_NY = 30, 22  # mismo grid que el dashboard para vectores

YEAR = 2005

# ── Endpoints NOAA PSL OPeNDAP ──
URL_SST_MON = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.mnmean.nc"
URL_SST_LTM = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2/sst.ltm.1991-2020.nc"
URL_SST_DAY = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc"
URL_U = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/uwnd.mon.mean.nc"
URL_V = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/vwnd.mon.mean.nc"
URL_T = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc"
URL_P = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.derived/surface/pres.mon.mean.nc"

# Track de Katrina (mismo del dashboard, fechas reales NHC best track)
KATRINA_TRACK = [
    {"date": "2005-08-25", "lon": -80.3, "lat": 23.1, "cat": 1},
    {"date": "2005-08-25", "lon": -82.6, "lat": 24.5, "cat": 1},
    {"date": "2005-08-26", "lon": -85.0, "lat": 24.4, "cat": 3},
    {"date": "2005-08-27", "lon": -87.2, "lat": 24.8, "cat": 4},
    {"date": "2005-08-28", "lon": -88.6, "lat": 25.4, "cat": 5},
    {"date": "2005-08-28", "lon": -89.6, "lat": 27.2, "cat": 5},
    {"date": "2005-08-29", "lon": -89.2, "lat": 28.5, "cat": 3},
    {"date": "2005-08-29", "lon": -88.8, "lat": 30.2, "cat": 1},
]


def log(msg):
    print(f"[fetch_katrina_data] {msg}", flush=True)


def to_pm180(ds, lon_name="lon"):
    """Reordena longitudes de 0-360 a -180..180 y ordena."""
    ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
    return ds.sortby(lon_name)


def regrid_bilinear(da, lon_target, lat_target, lon_name="lon", lat_name="lat"):
    """Interpola un DataArray 2D (lat,lon) a la grilla (lat_target, lon_target)."""
    out = da.interp({lon_name: lon_target, lat_name: lat_target}, method="linear")
    return out.values


def slice_region(ds, pad=2.0, lon_name="lon", lat_name="lat"):
    """Auto-detecta orientación de latitud y aplica el slice correcto."""
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    lat_asc = lat_vals[0] < lat_vals[-1]
    if lat_asc:
        lat_slice = slice(REGION["lat0"] - pad, REGION["lat1"] + pad)
    else:
        lat_slice = slice(REGION["lat1"] + pad, REGION["lat0"] - pad)
    lon_asc = lon_vals[0] < lon_vals[-1]
    lon_slice = slice(REGION["lon0"] - pad, REGION["lon1"] + pad) if lon_asc else slice(REGION["lon1"] + pad, REGION["lon0"] - pad)
    return ds.sel({lon_name: lon_slice, lat_name: lat_slice})


# Alias compat
slice_region_asc = slice_region


def safe_open(url, **kwargs):
    """Apertura tolerante: prueba engine netcdf4 y luego pydap si falla."""
    last_err = None
    for engine in (None, "netcdf4", "pydap"):
        try:
            log(f"  Abriendo OPeNDAP ({engine or 'auto'}): {url}")
            kw = {} if engine is None else {"engine": engine}
            return xr.open_dataset(url, **kw, **kwargs)
        except Exception as e:
            last_err = e
            log(f"    -> falló ({type(e).__name__}: {str(e)[:120]})")
    raise RuntimeError(f"No pude abrir {url}: {last_err}")


# ────────────────────────────────────────────────────────
#  CARGA DE DATOS
# ────────────────────────────────────────────────────────
def load_sst_monthly():
    log("Cargando SST mensual (NOAA OISST v2)...")
    ds = safe_open(URL_SST_MON)
    ds = to_pm180(ds)
    ds = slice_region_asc(ds)
    sst_2005 = ds.sst.sel(time=str(YEAR))
    log(f"  SST 2005 dims: {dict(sst_2005.sizes)}")
    return sst_2005


def load_sst_climatology():
    log("Cargando SST climatología 1991-2020 (NOAA OISST v2 LTM)...")
    try:
        ds = safe_open(URL_SST_LTM)
        ds = to_pm180(ds)
        ds = slice_region_asc(ds)
        # El archivo LTM tiene 12 meses
        sst_ltm = ds.sst
        if "time" in sst_ltm.dims:
            sst_ltm = sst_ltm
        return sst_ltm
    except Exception as e:
        log(f"  LTM no disponible ({e}); usando media 1991-2020 desde mnmean")
        ds = safe_open(URL_SST_MON)
        ds = to_pm180(ds)
        ds = slice_region_asc(ds)
        sst_all = ds.sst.sel(time=slice("1991-01-01", "2020-12-31"))
        clim = sst_all.groupby("time.month").mean("time")
        return clim


def load_ncep_var(url, varname, year):
    """Carga una variable NCEP/NCAR R1 mensual SOLO para el año pedido (rápido)."""
    log(f"Cargando {varname} (NCEP/NCAR R1) para {year}...")
    ds = safe_open(url)
    ds = to_pm180(ds)
    ds = slice_region(ds, pad=5)
    da_year = ds[varname].sel(time=str(year))
    return da_year, None  # no precomputamos climatología NCEP (no se usa)


def load_sst_daily_for_katrina():
    """Carga SST diaria de alta resolución (0.25°) para el track Katrina."""
    url = URL_SST_DAY.format(year=YEAR)
    log(f"Cargando SST diaria 0.25° {YEAR} (NOAA OISST v2 high-res)...")
    ds = safe_open(url)
    ds = to_pm180(ds)
    ds = slice_region_asc(ds, pad=4)
    # solo agosto 23-30 alrededor del peak Katrina
    sst_aug = ds.sst.sel(time=slice("2005-08-23", "2005-08-31"))
    return sst_aug


# ────────────────────────────────────────────────────────
#  PROCESADO Y EXPORTACIÓN
# ────────────────────────────────────────────────────────
def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)

    # Construir grilla de salida
    lon_target = np.linspace(REGION["lon0"], REGION["lon1"], NX, dtype=float)
    lat_target = np.linspace(REGION["lat0"], REGION["lat1"], NY, dtype=float)
    lon_wind = np.linspace(REGION["lon0"], REGION["lon1"], WIND_NX, dtype=float)
    lat_wind = np.linspace(REGION["lat0"], REGION["lat1"], WIND_NY, dtype=float)

    # 1. SST mensual + climatología
    sst_2005 = load_sst_monthly()  # (12, lat, lon)
    sst_clim = load_sst_climatology()  # (12, lat, lon) - month dim
    # Forzar carga en memoria
    log("  Descargando arrays SST a memoria...")
    sst_2005 = sst_2005.load()
    sst_clim = sst_clim.load()

    # Normalizar dim climatología
    clim_dim = "month" if "month" in sst_clim.dims else "time"
    if clim_dim == "time":
        # 12 valores -> renombrar a month
        sst_clim = sst_clim.rename({"time": "month"}).assign_coords(month=np.arange(1, 13))

    # 2. NCEP atm (sólo el año, sin climatología)
    u_2005, _ = load_ncep_var(URL_U, "uwnd", YEAR)
    v_2005, _ = load_ncep_var(URL_V, "vwnd", YEAR)
    t_2005, _ = load_ncep_var(URL_T, "air", YEAR)
    p_2005, _ = load_ncep_var(URL_P, "pres", YEAR)
    log("  Descargando NCEP a memoria...")
    for da in (u_2005, v_2005, t_2005, p_2005):
        da.load()

    # 3. SST diaria Katrina
    try:
        sst_day = load_sst_daily_for_katrina().load()
    except Exception as e:
        log(f"  No pude cargar SST diaria ({e}); track Katrina usará SST mensual")
        sst_day = None

    # ── Construir 12 meses para el dashboard ──
    months_out = []
    for mi in range(12):
        log(f"Procesando mes {mi+1}/12...")
        # SST 2005
        sst_m = sst_2005.isel(time=mi)
        sst_grid = regrid_bilinear(sst_m, lon_target, lat_target)
        # Climatología
        sst_c = sst_clim.sel(month=mi + 1)
        sst_clim_grid = regrid_bilinear(sst_c, lon_target, lat_target)
        # Anomalía
        anom_grid = sst_grid - sst_clim_grid

        # Viento (sobre grilla más gruesa)
        u_m = u_2005.isel(time=mi)
        v_m = v_2005.isel(time=mi)
        u_grid = regrid_bilinear(u_m, lon_wind, lat_wind)
        v_grid = regrid_bilinear(v_m, lon_wind, lat_wind)

        # T aire y presión (sobre grilla SST)
        t_m = t_2005.isel(time=mi)
        p_m = p_2005.isel(time=mi)
        t_grid = regrid_bilinear(t_m, lon_target, lat_target)
        p_grid = regrid_bilinear(p_m, lon_target, lat_target)
        # NCEP air viene en K -> °C ; pres en Pa (la dejamos en Pa, el HTML divide /100)
        if np.nanmean(t_grid) > 100:  # Kelvin
            t_grid = t_grid - 273.15

        # NaN handling — devuelve None (JSON null) para cualquier valor no finito
        def clean(arr):
            a = np.array(arr, dtype=float)
            return [None if not np.isfinite(v) else round(float(v), 3) for v in a.flatten()]

        def safe_float(v, decimals=3):
            """Convierte a float; devuelve None si no es finito."""
            try:
                f = float(v)
                return round(f, decimals) if np.isfinite(f) else None
            except Exception:
                return None

        months_out.append({
            "month": mi,
            "sst": clean(sst_grid),
            "sst_clim": clean(sst_clim_grid),
            "anom": clean(anom_grid),
            "u": clean(u_grid),
            "v": clean(v_grid),
            "tair": clean(t_grid),
            "pres": clean(p_grid),
        })

    # ── Extracción SST diaria sobre track Katrina ──
    katrina_real = []
    if sst_day is not None:
        for pt in KATRINA_TRACK:
            try:
                val = sst_day.sel(time=pt["date"]).interp(
                    lon=pt["lon"], lat=pt["lat"], method="linear"
                ).values
                v = float(val) if np.isfinite(val) else None
            except Exception:
                v = None
            # climatología para esta lat/lon en agosto (mes 8)
            try:
                clim_v = float(sst_clim.sel(month=8).interp(lon=pt["lon"], lat=pt["lat"]).values)
            except Exception:
                clim_v = None
            anom = (v - clim_v) if (v is not None and clim_v is not None) else None
            katrina_real.append({
                "date": pt["date"], "lon": pt["lon"], "lat": pt["lat"], "cat": pt["cat"],
                "sst": round(v, 2) if v is not None else None,
                "sst_clim": round(clim_v, 2) if clim_v is not None else None,
                "anom": round(anom, 2) if anom is not None else None,
            })

    # ── Serie temporal del punto central (40°N, -40°O) para chart ──
    timeseries = []
    for mi in range(12):
        # Buscar valor en grilla
        i = int(np.argmin(np.abs(lon_target - (-40))))
        j = int(np.argmin(np.abs(lat_target - 40)))
        idx = j * NX + i

        def _get(arr, fallback):
            v = arr[idx]
            if v is None: return fallback
            try:
                f = float(v)
                return f if np.isfinite(f) else fallback
            except Exception:
                return fallback

        sst_v  = _get(months_out[mi]["sst"],  14.0)
        anom_v = _get(months_out[mi]["anom"],  0.0)
        tair_v = _get(months_out[mi]["tair"], 12.0)

        # Z-score: usar std de la serie de 12 valores en ese punto
        sst_series = [m["sst"][idx] for m in months_out if m["sst"][idx] is not None]
        std  = float(np.std(sst_series))  if len(sst_series) > 1 else 1.0
        mean = float(np.mean(sst_series)) if sst_series else 0.0
        std  = std  if np.isfinite(std)  else 1.0
        mean = mean if np.isfinite(mean) else 0.0
        z = (sst_v - mean) / std if std > 0 else 0.0
        z = round(z, 3) if np.isfinite(z) else 0.0

        timeseries.append({
            "t":       mi,
            "sst":     round(sst_v,  3),
            "anom":    round(anom_v, 3),
            "airTemp": round(tair_v, 3),
            "zScore":  z,
        })

    # ── Métricas globales ──
    # Z-score global SST sobre toda la región-año
    all_sst = np.array([v for m in months_out for v in m["sst"] if v is not None], dtype=float)
    g_mean = float(np.mean(all_sst)) if len(all_sst) > 0 else 0.0
    g_std  = float(np.std(all_sst))  if len(all_sst) > 0 else 1.0
    # Garantizar que no son NaN/Inf
    g_mean = g_mean if np.isfinite(g_mean) else 0.0
    g_std  = g_std  if np.isfinite(g_std)  else 1.0

    out = {
        "meta": {
            "title": "Datos reales NOAA/NASA — Atlántico Norte 2005 (caso Huracán Katrina)",
            "sources": {
                "sst_monthly": "NOAA OISST v2 monthly mean (1°x1°)",
                "sst_climatology": "NOAA OISST v2 LTM 1991-2020",
                "sst_daily": "NOAA OISST v2 high-res daily (0.25°)",
                "wind_temp_pres": "NCEP/NCAR Reanalysis 1 monthly mean (2.5°)",
            },
            "note": (
                "DYAMOND-v2 (acoplado GEOS-MITgcm) cubre 2020-01-19 → 2020-03-01, "
                "por lo que para Katrina (Agosto 2005) usamos los productos NOAA/NCEP "
                "que SÍ cubren ese periodo. Esto es 'datos reales' de satélite y reanálisis."
            ),
            "region": REGION,
            "nx": NX, "ny": NY,
            "wind_nx": WIND_NX, "wind_ny": WIND_NY,
            "year": YEAR,
            "global_sst_mean": round(g_mean, 3),
            "global_sst_std": round(g_std, 3),
        },
        "months": months_out,
        "timeseries": timeseries,
        "katrina_track_real": katrina_real,
    }

    log(f"Escribiendo {OUT} ...")
    # json.dumps de Python escribe NaN/Infinity literales que NO son JSON válido.
    # Usamos allow_nan=False para detectar escapes y forzamos None en los arrays
    # mediante el helper clean() que ya devuelve None para np.nan.
    json_str = json.dumps(out, allow_nan=False)
    OUT.write_text(json_str, encoding="utf-8")
    sz_mb = OUT.stat().st_size / 1024 / 1024
    log(f"OK — {OUT.name} ({sz_mb:.2f} MB)")
    log(f"Resumen: {len(months_out)} meses, SST global mean {g_mean:.2f}°C "
        f"(σ={g_std:.2f}), {len(katrina_real)} puntos track Katrina con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
