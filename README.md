# 🌊 ClimaMacro-CL

> **Sistema MLOps de predicción macroeconómica basado en señales climáticas para Chile**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ¿Qué es ClimaMacro-CL?

ClimaMacro-CL relaciona fenómenos climáticos globales (El Niño / La Niña, medidos por el
índice ONI de NOAA) con indicadores macroeconómicos de Chile: IPC, precio del cobre y
precipitaciones en la Región Metropolitana.

El sistema ingesta datos reales de fuentes públicas, entrena modelos predictivos con
MLflow para tracking de experimentos, sirve predicciones vía FastAPI y monitorea
el drift del modelo mensualmente con reentrenamiento automático vía GitHub Actions.

---

## 🏗️ Arquitectura del sistema

```
Fuentes de datos                 Pipeline MLOps               Consumo
─────────────────               ────────────────             ─────────
NOAA  (ONI index)  ──┐
Banco Central API  ──┼──► Ingesta ──► EDA ──► Features ──► MLflow
DGA   (precipit.)  ──┘         │                               │
                               ▼                               ▼
                          data/processed            Entrenamiento / Registry
                                                               │
                                                               ▼
                                                    FastAPI /predict endpoint
                                                               │
                                                    ┌──────────┴──────────┐
                                                    ▼                     ▼
                                              Streamlit              GitHub Actions
                                             Dashboard            (reentrenamiento
                                              público               mensual + CI)
```

---

## 🗂️ Estructura del repositorio

```
ClimaMacro-CL/
├── data/
│   ├── raw/
│   │   ├── noaa/          # ONI index (ERSST v5)
│   │   ├── bcentral/      # IPC, precio cobre (Banco Central de Chile)
│   │   └── dga/           # Precipitaciones DGA
│   ├── processed/         # Datos limpios y mergeados
│   └── external/          # Datasets auxiliares
├── notebooks/             # EDA y experimentos exploratorios
├── src/
│   ├── ingestion/         # Clientes para cada fuente de datos
│   ├── processing/        # Limpieza, normalización, merge temporal
│   ├── features/          # Feature engineering (lags, rolling means)
│   ├── models/            # Entrenamiento XGBoost, Prophet, LSTM
│   ├── serving/           # FastAPI endpoint /predict
│   ├── monitoring/        # Drift detection y alertas
│   └── dashboard/         # Streamlit app
├── tests/
│   ├── unit/              # Tests por módulo
│   └── integration/       # Tests end-to-end
├── .github/workflows/     # CI/CD y reentrenamiento mensual
├── docker/                # Dockerfiles por servicio
├── configs/               # Parámetros del modelo y del pipeline
├── mlruns/                # Artefactos MLflow (excluidos de Git)
├── logs/                  # Logs estructurados (excluidos de Git)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Fases del proyecto

| Fase | Días | Estado | Descripción |
|------|------|--------|-------------|
| **1 — Datos y EDA** | 1–21 | 🔄 En progreso | Ingesta NOAA, Banco Central, DGA · Limpieza · Correlaciones |
| **2 — Modelado** | 22–42 | ⏳ Pendiente | Feature engineering · MLflow · XGBoost + Prophet/LSTM · Walk-forward CV |
| **3 — MLOps & Deploy** | 43–70 | ⏳ Pendiente | Docker · FastAPI · GitHub Actions · GCP Cloud Run · Drift detection |
| **4 — Observabilidad** | 71–91 | ⏳ Pendiente | Dashboard Streamlit · Monitoreo · README portafolio final |

---

## ⚙️ Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/GabrielVerdejo/ClimaMacro-CL.git
cd ClimaMacro-CL

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -c "import pandas, xgboost, mlflow; print('OK')"
```

---

## 📡 Fuentes de datos

| Fuente | Dataset | URL | Frecuencia |
|--------|---------|-----|------------|
| NOAA / CPC | ONI Index (ERSST v5) | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt | Mensual |
| Banco Central de Chile | IPC, Cobre, Tipo de cambio | https://si3.bcentral.cl/Siete/ | Mensual |
| DGA (MOP) | Precipitaciones RM | https://dga.mop.gob.cl/servicioshidrometeorologicos/ | Diario → Mensual |

---

## 🧪 Tests

```bash
# Todos los tests
pytest tests/ -v

# Solo unitarios
pytest tests/unit/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Autor

**Gabriel Verdejo**  
Backend Developer @ BCI (vía Neoris) · ML Engineer en formación  
Python · Java/Spring Boot · Azure · GCP · MLflow · FastAPI  

---

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).
