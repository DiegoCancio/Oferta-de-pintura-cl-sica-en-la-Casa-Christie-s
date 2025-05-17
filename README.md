E# Análisis del mercado del arte clásico — Proyecto final del bootcamp

Este trabajo pone en práctica todas las destrezas adquiridas durante el bootcamp de Data Analytics. El objetivo es analizar el mercado del arte a través de las subastas de pintura clásica organizadas por la casa Christie's desde 2017.

---

## 📦 Estructura del proyecto

### 1. `ChristiesGrandesMaestrosWbScrp.ipynb`
Notebook de **web scraping**:
- Extrae información directamente de la web de Christie's sobre subastas de pintura clásica desde 2017.
- Contiene mucho trabajo previo de limpieza y pruebas antes de llegar al scraping funcional.
- **Producto final**: `datos_subastas_final3trasrepesca.csv`, con los datos esenciales ya recopilados.

### 2. `ChristiesGrandesMaestrosProcesamiento.ipynb`
Notebook de **procesamiento y limpieza**:
- Esta parte es especialmente compleja debido a la estructura no estructurada de los textos.
- Se extraen datos clave como:
  - Nombre, país y fechas del autor
  - Técnica, soporte, dimensiones
  - Tipología (cuadro, estatua, mueble, etc.)
  - Precio de venta (si aplica)
- También se detectan y corrigen **duplicidades** (como variaciones en nombres de artistas).
- **Archivos generados**:
  - `datos_subastas_procesado.csv`
  - `Artistas.csv`
  - `Subastas.csv`

### 3. `ChrtGM_EDA.ipynb`
Notebook de **análisis exploratorio (EDA)**:
- Análisis clásico con gráficos y estadísticas sobre los datasets ya limpios.
- Uso de `ipywidgets` para crear paneles **interactivos**, permitiendo explorar:
  - Años
  - Países
  - Categorías
  - Variables de obras subastadas

### 4. `app.py`
Aplicación final con **Streamlit**:
- Visualización interactiva del mercado del arte.
- Permite navegar intuitivamente por los datos.
- Ideal para cualquier usuario interesado en explorar sin necesidad de código.

---

## 🛠️ Tecnologías y herramientas usadas

- Python (numpy, pandas, matplotlib, seaborn, plotly, selenium, ipywidgets, Streamlit, etc.)
- Web Scraping
- Limpieza de datos complejos no estructurados
- Visualización interactiva

---

## 📎 Notas finales

Todo el trabajo fue realizado de forma individual como proyecto final del bootcamp.  
Las etapas reflejan el flujo de un proyecto real de análisis de datos: desde la obtención bruta hasta la creación de una herramienta de consulta final.

