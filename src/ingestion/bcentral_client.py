"""
Cliente async para descarga y parseo de series del Banco Central de Chile (BDE).

Descarga series macroeconómicas vía la API REST del portal SIETE:
    - IPC (Índice de Precios al Consumidor) — mensual
    - Precio del cobre (onza troy, USD) — diario, se agrega a mensual

Todas las series se resamplean a frecuencia mensual (primer día del mes) para
permitir el merge temporal posterior con el ONI. Series ya mensuales quedan
inalteradas; series diarias se promedian por mes.

Fuente: https://si3.bcentral.cl/SieteRestWS/
Requiere credenciales (registro gratuito en https://si3.bcentral.cl/Siete/).
Las credenciales se leen desde el archivo .env (BCENTRAL_USER, BCENTRAL_PASS).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
BCENTRAL_BASE_URL: str = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"

# Códigos de serie (confirmados en el buscador de series BDE)
# IPC: variación mensual (%) — serie estacionaria, ideal para correlación con ONI
IPC_SERIES_ID: str = "F074.IPC.VAR.Z.Z.C.M"
COBRE_SERIES_ID: str = "F019.PPB.PRE.100.D"

# Mapeo de código de serie → nombre de columna limpio
SERIES_MAP: dict[str, str] = {
    IPC_SERIES_ID: "ipc",
    COBRE_SERIES_ID: "cobre",
}

PROCESSED_PATH: Path = Path("data/processed/bcentral.parquet")

# Rango temporal por defecto (alineado con el ONI)
DEFAULT_FIRST_DATE: str = "1990-01-01"

TIMEOUT_SECONDS: int = 30

# Flag de la API para valores no disponibles
NO_DATA_FLAG: str = "ND"

# Frecuencia objetivo del merge (Month Start)
TARGET_FREQUENCY: str = "MS"


# ---------------------------------------------------------------------------
# Credenciales
# ---------------------------------------------------------------------------
def load_credentials() -> tuple[str, str]:
    """
    Carga las credenciales del Banco Central desde el archivo .env.

    Returns:
        Tupla (usuario, password).

    Raises:
        ValueError: Si alguna de las credenciales no está definida en .env.
    """
    load_dotenv()

    user = os.getenv("BCENTRAL_USER")
    password = os.getenv("BCENTRAL_PASS")

    if not user or not password:
        raise ValueError(
            "Credenciales no encontradas. Definí BCENTRAL_USER y BCENTRAL_PASS "
            "en el archivo .env (registro en https://si3.bcentral.cl/Siete/)."
        )

    return user, password


# ---------------------------------------------------------------------------
# Resampling a mensual
# ---------------------------------------------------------------------------
def resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una serie a frecuencia mensual (promedio por mes).

    Series ya mensuales quedan prácticamente inalteradas (un valor por mes).
    Series diarias se promedian dentro de cada mes calendario.

    Args:
        df: DataFrame con columnas date (datetime) y value (float).

    Returns:
        DataFrame con una fila por mes, indexado al primer día del mes.
    """
    monthly = (
        df.set_index("date")
        .resample(TARGET_FREQUENCY)["value"]
        .mean()
        .reset_index()
    )
    return monthly


# ---------------------------------------------------------------------------
# Descarga de una serie
# ---------------------------------------------------------------------------
async def fetch_series(
    client: httpx.AsyncClient,
    series_id: str,
    user: str,
    password: str,
    first_date: str = DEFAULT_FIRST_DATE,
) -> pd.DataFrame:
    """
    Descarga, parsea y agrega a mensual una serie individual del Banco Central.

    Args:
        client: Cliente httpx async reutilizable.
        series_id: Código de la serie en la API BDE.
        user: Usuario del Banco Central.
        password: Password del Banco Central.
        first_date: Fecha inicial en formato YYYY-MM-DD.

    Returns:
        DataFrame con columnas:
            - date (datetime64[ns]): primer día del mes.
            - value (float64): valor mensual (promedio si la serie era diaria).

    Raises:
        httpx.HTTPStatusError: Si la API responde con un código de error.
        ValueError: Si la respuesta no tiene el formato esperado.
    """
    params = {
        "user": user,
        "pass": password,
        "function": "GetSeries",
        "timeseries": series_id,
        "firstdate": first_date,
    }

    logger.info(f"Descargando serie {series_id} desde Banco Central")

    response = await client.get(
        BCENTRAL_BASE_URL, params=params, timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()

    # La API del Banco Central responde en Latin-1 (acentos españoles en las
    # descripciones), no en UTF-8. Decodificamos explícitamente antes de parsear.
    payload = json.loads(response.content.decode("latin-1"))

    # La API devuelve Codigo=0 cuando la petición fue exitosa
    codigo = payload.get("Codigo")
    if codigo != 0:
        descripcion = payload.get("Descripcion", "sin descripción")
        raise ValueError(
            f"Error de la API para {series_id}: código {codigo} — {descripcion}"
        )

    observations = payload.get("Series", {}).get("Obs", [])
    if not observations:
        raise ValueError(f"La serie {series_id} no devolvió observaciones.")

    # Parsear observaciones: fecha en dd-mm-yyyy, valor como string
    records: list[dict] = []
    for obs in observations:
        raw_value = obs.get("value", NO_DATA_FLAG)

        # Convertir valor; ND (no disponible) se marca como nulo
        value = None if raw_value == NO_DATA_FLAG else float(raw_value)

        date = pd.to_datetime(obs["indexDateString"], format="%d-%m-%Y")
        records.append({"date": date, "value": value})

    df = pd.DataFrame(records, columns=["date", "value"])
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    raw_count = len(df)

    # Agregar a mensual (promedio por mes)
    df = resample_to_monthly(df)

    valid_count = df["value"].notna().sum()
    logger.success(
        f"Serie {series_id}: {raw_count} obs. crudas → {len(df)} meses "
        f"({valid_count} válidos, "
        f"{df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')})"
    )

    return df


# ---------------------------------------------------------------------------
# Descarga de todas las series
# ---------------------------------------------------------------------------
async def fetch_all_series(
    first_date: str = DEFAULT_FIRST_DATE,
) -> pd.DataFrame:
    """
    Descarga todas las series configuradas y las combina en un solo DataFrame.

    Las series se descargan concurrentemente (async), se agregan a mensual
    individualmente y se mergean por fecha.

    Args:
        first_date: Fecha inicial en formato YYYY-MM-DD.

    Returns:
        DataFrame con columnas:
            - date (datetime64[ns])
            - ipc (float64)
            - cobre (float64)
    """
    user, password = load_credentials()

    async with httpx.AsyncClient() as client:
        # Lanzar las descargas concurrentemente
        tasks = [
            fetch_series(client, series_id, user, password, first_date)
            for series_id in SERIES_MAP
        ]
        results = await asyncio.gather(*tasks)

    # Renombrar la columna value de cada DataFrame al nombre de la serie
    dataframes: list[pd.DataFrame] = []
    for series_id, df in zip(SERIES_MAP.keys(), results):
        column_name = SERIES_MAP[series_id]
        dataframes.append(df.rename(columns={"value": column_name}))

    # Merge outer por fecha para conservar todas las observaciones
    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    logger.success(
        f"Series combinadas: {len(merged)} filas, columnas {list(merged.columns)}"
    )

    return merged


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------
def save_bcentral_parquet(
    df: pd.DataFrame,
    processed_path: Path = PROCESSED_PATH,
) -> Path:
    """
    Guarda el DataFrame combinado como archivo Parquet.

    Args:
        df: DataFrame con las series del Banco Central.
        processed_path: Ruta de destino del archivo .parquet.

    Returns:
        Ruta al archivo guardado.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path, index=False)
    logger.success(f"Series guardadas como Parquet en {processed_path}")
    return processed_path


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
async def run_bcentral_pipeline(
    first_date: str = DEFAULT_FIRST_DATE,
    processed_path: Path = PROCESSED_PATH,
) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo: descarga → agregación mensual → merge → guardado.

    Args:
        first_date: Fecha inicial en formato YYYY-MM-DD.
        processed_path: Ruta para el archivo Parquet procesado.

    Returns:
        DataFrame con las series combinadas.
    """
    logger.info("=== Iniciando pipeline Banco Central ===")

    df = await fetch_all_series(first_date=first_date)
    save_bcentral_parquet(df=df, processed_path=processed_path)

    logger.info("=== Pipeline Banco Central completado ===")
    return df


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True)

    df_bcentral = asyncio.run(run_bcentral_pipeline())

    print("\n--- Primeros registros ---")
    print(df_bcentral.head(10).to_string(index=False))

    print("\n--- Últimos registros ---")
    print(df_bcentral.tail(5).to_string(index=False))

    print("\n--- Info de nulos ---")
    print(df_bcentral.isna().sum().to_string())