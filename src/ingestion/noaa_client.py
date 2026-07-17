"""
Cliente para descarga y parseo del Oceanic Niño Index (ONI) desde NOAA/CPC.

El ONI es el estándar oficial para clasificar episodios El Niño / La Niña.
Fuente: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
Formato: media móvil de 3 meses de anomalías SST en la región Niño 3.4 (ERSSTv5).
Formato del archivo: largo (SEAS, YR, TOTAL, ANOM) — una fila por temporada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ONI_URL: str = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Rutas relativas a la raíz del proyecto
RAW_PATH: Path = Path("data/raw/noaa/oni_raw.txt")
PROCESSED_PATH: Path = Path("data/processed/oni.parquet")

# Mes central de cada temporada de 3 meses
SEASON_TO_MONTH: dict[str, int] = {
    "DJF": 1,   # Dic-Ene-Feb  → mes central Enero
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,  # Nov-Dic-Ene  → mes central Diciembre
}

TIMEOUT_SECONDS: int = 30

# NOAA bloquea requests sin User-Agent; usamos uno estándar de navegador
REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ClimaMacro-CL/0.1; "
        "https://github.com/gabrielcamilov/ClimaMacro-CL)"
    )
}


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def download_oni_raw(
    url: str = ONI_URL,
    raw_path: Path = RAW_PATH,
) -> Path:
    """
    Descarga el archivo ONI crudo desde NOAA y lo guarda en disco.

    Args:
        url: URL del archivo oni.ascii.txt en NOAA/CPC.
        raw_path: Ruta local donde guardar el archivo descargado.

    Returns:
        Ruta al archivo descargado.

    Raises:
        requests.HTTPError: Si NOAA responde con un código de error HTTP.
        requests.ConnectionError: Si no hay conexión a internet.
    """
    logger.info(f"Descargando ONI index desde {url}")

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    raw_path.write_text(response.text, encoding="utf-8")
    logger.success(f"Archivo crudo guardado en {raw_path} ({len(response.text):,} bytes)")

    return raw_path


# ---------------------------------------------------------------------------
# Parseo
# ---------------------------------------------------------------------------
def parse_oni(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """
    Parsea el archivo oni.ascii.txt al formato de serie temporal mensual.

    NOAA usa formato largo con columnas: SEAS, YR, TOTAL, ANOM.
    Cada fila representa una temporada de 3 meses (DJF, JFM, ..., NDJ).
    Este método mapea cada temporada al mes central y construye una fecha.

    Args:
        raw_path: Ruta al archivo crudo descargado de NOAA.

    Returns:
        DataFrame con columnas:
            - date (datetime64[ns]): primer día del mes central de cada temporada.
            - oni (float64): anomalía ONI en grados Celsius.

    Raises:
        FileNotFoundError: Si el archivo crudo no existe en la ruta indicada.
        ValueError: Si el archivo no tiene el formato esperado.
    """
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Archivo crudo no encontrado: {raw_path}. "
            "Ejecutá download_oni_raw() primero."
        )

    logger.info(f"Parseando ONI desde {raw_path}")

    # Leer el archivo — espacios variables como separador
    df_raw = pd.read_csv(
        raw_path,
        sep=r"\s+",
        engine="python",
    )

    logger.debug(f"Columnas del archivo fuente: {list(df_raw.columns)}")
    logger.debug(f"Filas encontradas: {len(df_raw)}")

    # Validar columnas del formato largo (SEAS, YR, TOTAL, ANOM)
    required_columns = {"SEAS", "YR", "ANOM"}
    missing = required_columns - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Formato inesperado: faltan columnas {missing}. "
            f"Columnas encontradas: {list(df_raw.columns)}"
        )

    # Filtrar valores nulos (-99.9 es el flag de NOAA para datos faltantes)
    df_raw = df_raw[df_raw["ANOM"] != -99.9].copy()

    # Mapear temporada al mes central
    df_raw["month"] = df_raw["SEAS"].map(SEASON_TO_MONTH)

    # Advertir si hay temporadas no reconocidas
    unknown_seasons = df_raw[df_raw["month"].isna()]["SEAS"].unique()
    if len(unknown_seasons) > 0:
        logger.warning(f"Temporadas no reconocidas (se ignorarán): {unknown_seasons}")
        df_raw = df_raw[df_raw["month"].notna()]

    # Construir columna date como primer día del mes central
    df_raw["date"] = pd.to_datetime(
        df_raw["YR"].astype(int).astype(str)
        + "-"
        + df_raw["month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )

    # Seleccionar y limpiar columnas finales
    df = (
        df_raw[["date", "ANOM"]]
        .rename(columns={"ANOM": "oni"})
        .drop_duplicates(subset="date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    df["oni"] = df["oni"].astype(float)

    logger.success(
        f"ONI parseado: {len(df)} registros "
        f"({df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')})"
    )

    return df


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def save_oni_parquet(
    df: pd.DataFrame,
    processed_path: Path = PROCESSED_PATH,
) -> Path:
    """
    Guarda el DataFrame del ONI como archivo Parquet.

    Args:
        df: DataFrame con columnas date y oni.
        processed_path: Ruta de destino del archivo .parquet.

    Returns:
        Ruta al archivo guardado.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.success(f"ONI guardado como Parquet en {processed_path}")
    return processed_path


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def run_oni_pipeline(
    url: str = ONI_URL,
    raw_path: Path = RAW_PATH,
    processed_path: Path = PROCESSED_PATH,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo: descarga → parseo → guardado.

    Args:
        url: URL del ONI en NOAA/CPC.
        raw_path: Ruta para el archivo crudo.
        processed_path: Ruta para el archivo Parquet procesado.

    Returns:
        DataFrame con la serie temporal ONI lista para usar.
    """
    logger.info("=== Iniciando pipeline ONI ===")

    raw_path = download_oni_raw(url=url, raw_path=raw_path)
    df = parse_oni(raw_path=raw_path)
    save_oni_parquet(df=df, processed_path=processed_path)

    logger.info("=== Pipeline ONI completado ===")
    return df


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True)

    df_oni = run_oni_pipeline()

    print("\n--- Primeros registros ---")
    print(df_oni.head(10).to_string(index=False))

    print("\n--- Últimos registros ---")
    print(df_oni.tail(5).to_string(index=False))

    print("\n--- Estadísticas ---")
    print(df_oni["oni"].describe().round(3).to_string())