import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime


st.set_page_config(
    page_title="Christie's Art Analysis",
    layout="wide",
)

def traducir_ciudad(ciudad):
    traducciones = {
        "LONDON": "Londres",
        "PARIS": "París",
        "NEW YORK": "Nueva York"
    }
    return traducciones.get(ciudad.upper(), ciudad)

def formatear_fecha(fecha):
    try:
        fecha_dt = datetime.strptime(fecha, '%Y-%m-%d')
        return fecha_dt.strftime('%-d de %B de %Y')
    except ValueError:
        return fecha

def formatear_precio(precio, ciudad):
    moneda = {"Londres": "£", "París": "€", "Nueva York": "$"}
    ciudad_traducida = traducir_ciudad(ciudad)
    simbolo = moneda.get(ciudad_traducida, "")
    return f"{simbolo}{precio:,.0f}".replace(",", " ") 

st.sidebar.title("Navegación")
page = st.sidebar.radio("Elige la página", ["Introducción y Exposiciones", "Datos de Subastas", "Modelo de ML"])

if page == "Introducción y Exposiciones":
   
    st.title("Análisis de Obras de Arte Subastadas en Christie's")
    st.markdown("""
    Esta aplicación permite explorar datos de obras de arte clásicas subastadas en Christie's. 
    Usaremos distintos filtros para visualizar las obras y análisis según variables como País, Año de Nacimiento del Artista, Nombre y Género.
    """)

    st.markdown(
        """
        <style>
        .img-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 400px;
            border: 1px solid #ddd;
            overflow: hidden;
        }
        .img-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: scale-down;
        }
        .img-text {
            text-align: center;
            margin: 10px 0;
        }
        .button-container {
            display: flex;
            justify-content: center;  /* Centrar los botones */
            width: 100%;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)


    df_works = pd.read_csv(r'C:\Users\34699\OneDrive\Escritorio\ProyectoFinalSubastas\SubastasArteClasico\Datos\datos_subastas_procesado.csv')

    df_works = df_works.dropna(subset=['URL imagen', 'Nombre', 'País', 'Género', 'Año de nacimiento', 'EstA'])

    if 'img_index' not in st.session_state:
        st.session_state.img_index = 0

    col1, col2 = st.columns([1, 3])

    with col1:

        st.header("Filtros")

        min_year = int(df_works['Año de nacimiento'].min())
        max_year = int(df_works['Año de nacimiento'].max())
        anio_range = st.slider(
            "Seleccione el rango de año de nacimiento",
            min_year, max_year, (min_year, max_year)
        )

        filtered_df = df_works[df_works['Año de nacimiento'].between(anio_range[0], anio_range[1])]

        precio_min = df_works['EstA'].min() if not df_works['EstA'].isnull().all() else 0
        precio_max = df_works['EstA'].max()


        if precio_min > 0:
            log_precio_min = np.log(precio_min)
        else:
            log_precio_min = 0 

        log_precio_max = np.log(precio_max)

        precio_range = st.slider(
            "Seleccione el rango de precio estimado (en el logaritmo de los precios)",
            float(log_precio_min), float(log_precio_max), (float(log_precio_min), float(log_precio_max)),
        )
        
        lower_bound = np.exp(precio_range[0]) 
        upper_bound = np.exp(precio_range[1]) 
        filtered_df = filtered_df[
            filtered_df['EstA'].between(lower_bound, upper_bound)
        ]

        pais_filtrado = ["Todos"] + list(filtered_df['País'].unique())
        pais = st.selectbox("Seleccione el País de Origen del Artista", options=pais_filtrado)

        if pais != "Todos":
            filtered_df = filtered_df[filtered_df['País'] == pais]

        nombre_filtrado = ["Todos"] + sorted(list(filtered_df['Nombre'].unique())) 
        nombre = st.selectbox("Seleccione el Nombre del Artista", options=nombre_filtrado)

        if nombre != "Todos":
            filtered_df = filtered_df[filtered_df['Nombre'] == nombre]

        genero_filtrado = ["Todos"] + list(filtered_df['Género'].unique())
        genero = st.selectbox("Seleccione el Género", options=genero_filtrado)

        if genero != "Todos":
            filtered_df = filtered_df[filtered_df['Género'] == genero]

        st.markdown(f"**Número de obras:** {len(filtered_df)}")  

    with col2:
        if not filtered_df.empty:
            img_index = st.session_state.img_index
            obra = filtered_df.iloc[img_index]
            img_url = obra['URL imagen']
            nombre = obra['Nombre']
            genero = obra['Género']
            pais_artista = obra['País']
            año_nacimiento = int(obra['Año de nacimiento'])
            ciudad = traducir_ciudad(obra['Ciudad'])
            fecha = formatear_fecha(obra['Fecha'])
            precio_num = obra['Precio_num']
            
            venta_info = f"Subastada en {ciudad} el {fecha}<br>"
            if pd.isna(precio_num):
                venta_info += "No vendida"
            else:
                venta_info += f"Vendida por {formatear_precio(precio_num, ciudad)}"

            st.markdown(f'<div class="img-container"><img src="{img_url}" alt="{nombre} - {genero}" /></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="img-text">{nombre} - {genero}<br>{pais_artista} | {año_nacimiento}<br>{venta_info}</div>', unsafe_allow_html=True)
            
        else:
            st.write("No se encontraron imágenes con los filtros seleccionados.")

        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        if st.button("Anterior"):
            if st.session_state.img_index > 0:  
                st.session_state.img_index -= 1

        if st.button("Siguiente"):
            if st.session_state.img_index < len(filtered_df) - 1: 
                st.session_state.img_index += 1
        
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Datos de Subastas":
    st.title("Datos de Subastas")
    st.markdown("""
    Aquí se pueden explorar los datos de las subastas, ver análisis estadísticos y realizar visualizaciones para entender mejor el mercado de arte.
    """)

    df_auctions = pd.DataFrame({
        "obra": ["Obra 1", "Obra 2"],
        "precio_venta": [1000000, 2000000],
        "fecha_subasta": ["2023-01-01", "2023-06-01"],
        "ciudad": ["LONDON", "NEW YORK"] 
    })

    st.write(df_auctions) 

    st.write("### Estadísticas de Precios de Venta")
    st.write(df_auctions.describe())

elif page == "Modelo de ML":
    st.title("Modelo de ML para Predecir Ventas")
    st.markdown("""
    En esta sección se despliega un modelo de Machine Learning que predice las ventas de las subastas según los datos históricos.
    """)

    st.write("Próximamente...")

