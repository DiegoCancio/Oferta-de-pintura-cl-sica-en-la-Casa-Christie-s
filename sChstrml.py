import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Christie's Art Analysis",
    layout="wide",
)

# Barra lateral para la navegación
st.sidebar.title("Navegación")
page = st.sidebar.selectbox("Elige la página", ["Introducción y Exposiciones", "Datos de Subastas", "Modelo de ML"])

if page == "Introducción y Exposiciones":
    st.title("Análisis de Obras de Arte Subastadas en Christie's")
    st.markdown("""
    Esta aplicación permite explorar datos de obras de arte clásicas subastadas en Christie's. 
    Usaremos distintos filtros para visualizar las obras y análisis según variables como País y Año de Nacimiento de los Artistas.
    """)

    # Sección de Presentación de Fotos y Filtros
    st.header("Exposición de Obras")
    
    # Aquí debes reemplazar esto con tu DataFrame de obras
    # df_works debe contener al menos las columnas: 'url_foto', 'pais', 'anio_nacimiento'
    df_works = pd.DataFrame({
        "url_foto": ["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"],
        "pais": ["España", "Francia"],
        "anio_nacimiento": [1746, 1826]
    })

    # Filtros
    pais = st.selectbox("Seleccione el País de Origen del Artista", df_works['pais'].unique())
    anio = st.selectbox("Seleccione el Año de Nacimiento del Artista", sorted(df_works['anio_nacimiento'].unique()))

    # Filtrar DataFrame
    filtered_df = df_works[(df_works['pais'] == pais) & (df_works['anio_nacimiento'] == anio)]

    # Mostrar fotos filtradas
    for index, row in filtered_df.iterrows():
        st.image(row['url_foto'], caption=f"Artista nacido en {anio} de {pais}", use_column_width=True)

elif page == "Datos de Subastas":
   st.title("Datos de Subastas")
   st.markdown("""
   Aquí se pueden explorar los datos de las subastas, ver análisis estadísticos y realizar visualizaciones para entender mejor el mercado de arte.
   """)

   # Aquí debes reemplazar esto con tu DataFrame de subastas
   # df_auctions debe contener al menos columnas significativas para análisis
   df_auctions = pd.DataFrame({
       "obra": ["Obra 1", "Obra 2"],
       "precio_venta": [1000000, 2000000],
       "fecha_subasta": ["2023-01-01", "2023-06-01"]
   })

   st.write(df_auctions)  # Mostrar DataFrame de subastas

   st.write("### Estadísticas de Precios de Venta")
   st.write(df_auctions.describe())

elif page == "Modelo de ML":
    st.title("Modelo de ML para Predecir Ventas")
    st.markdown("""
    En esta sección se despliega un modelo de Machine Learning que predice las ventas de las subastas según los datos históricos.
    """)

    st.write("Próximamente...")
