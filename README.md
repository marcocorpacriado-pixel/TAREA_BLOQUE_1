# TAREA_BLOQUE_1

# 📈 Extractor de Datos Financieros Multi-API

Este proyecto forma parte del Máster en **Inteligencia Artificial y Computación Cuántica Aplicada a los Mercados Financieros**  
Su objetivo es desarrollar un **programa extractor de información financiera** desde distintas APIs online, garantizando:
- Un **formato de salida estandarizado** (igual sin importar la fuente)
- Posibilidad de **descargar distintas tipologías de datos** (acciones, índices, divisas…)
- Capacidad de **extraer N series a la vez**

---

## 🧩 Estructura del proyecto
src/
└── app/
├── main.py # Programa principal (CLI)
└── extractor/ # Módulos de conexión con cada API
├── base.py # Clase base y dataclass 'Candle'
├── alphavantage.py # Fuente 1: Alpha Vantage (acciones + FX)
├── marketstack.py # Fuente 2: MarketStack (acciones)
└── twelvedata.py # Fuente 3: TwelveData (acciones + FX)
.env.example # Variables de entorno (API keys)
requirements.txt # Dependencias necesarias


## ⚙️ Instalación y configuración

### 1️) CLONA EL REPOSITORIO
git clone https://github.com/tu_usuario/TAREA_BLOQUE_1.git
cd TAREA_BLOQUE_1
### 2) INSTALA DEPENDENCIAS
pip install -r requirements.txt
### 3) CREA TU ARCHIVO .env
ALPHAVANTAGE_API_KEY=tu_clave_aqui
MARKETSTACK_API_KEY=tu_clave_aqui
TWELVEDATA_API_KEY=tu_clave_aqui

### FUNCIONAMIENTO DEL EXTRACTOR
El programa puede extraer información desde varias fuentes, manteniendo el mismo formato de salida para todos los casos.
Tipo de datos soportados: 
-prices --> Datos históricos de acciones, índices, ETFs
-fx --> Datos históricos de pares de divisas 

💻 Uso desde terminal o Google Colab
🪙 1. Descargar varias acciones a la vez (AlphaVantage)
python -m app.main fetch --provider alphavantage --datatype prices \
  --symbols AAPL,MSFT,SPY --start 2024-01-01 --end 2024-06-30 --format csv


📂 Generará:

data/alphavantage/prices/AAPL.csv
data/alphavantage/prices/MSFT.csv
data/alphavantage/prices/SPY.csv

💱 2. Descargar datos de divisas (FX)
python -m app.main fetch --provider alphavantage --datatype fx \
  --symbols EURUSD,USDJPY --start 2024-01-01 --end 2024-03-31 --format parquet


📂 Generará:

data/alphavantage/fx/EURUSD.parquet
data/alphavantage/fx/USDJPY.parquet

📁 3. Descargar N series desde un archivo

Crea un fichero symbols.txt con un símbolo por línea:

AAPL
MSFT
SPY


Y ejecuta:

python -m app.main fetch --provider alphavantage --datatype prices \
  --symbols-file symbols.txt --format parquet

📊 Formato estandarizado de salida

Todos los proveedores generan exactamente el mismo formato de columnas:

provider	datatype	symbol	date	open	high	low	close	volume
alphavantage	prices	AAPL	2024-01-02	185.6	188.2	184.9	187.9	56930000

De este modo, los datos pueden combinarse fácilmente en Power BI, Excel, R o Python para análisis posteriores.

🚀 Opciones de ejecución (CLI)
Parámetro	Descripción	Ejemplo
--provider	API a usar (alphavantage, marketstack, twelvedata)	--provider twelvedata
--datatype	Tipo de datos (prices, fx)	--datatype fx
--symbol	Símbolo único	--symbol AAPL
--symbols	Lista separada por comas	--symbols AAPL,MSFT,SPY
--symbols-file	Archivo con símbolos	--symbols-file symbols.txt
--start / --end	Rango de fechas (YYYY-MM-DD)	--start 2024-01-01 --end 2024-06-30
--format	Formato de salida (csv, parquet)	--format parquet
--outdir	Carpeta base para los datos	--outdir data
📦 Dependencias

requests

python-dotenv

pandas (solo si usas formato Parquet)

