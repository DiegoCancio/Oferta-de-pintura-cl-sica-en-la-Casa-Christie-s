import streamlit as st
import pandas as pd
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re  
import requests
from bs4 import BeautifulSoup
from matplotlib.patches import Circle

import unicodedata
from PIL import Image
import requests
from io import BytesIO
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from io import BytesIO
import threading
from IPython.display import display

import plotly.express as px
import ipywidgets as widgets
from ipywidgets import interact, VBox, HBox
import pandas as pd
import ipywidgets as widgets
from ipywidgets import HTML, VBox, HBox, Dropdown, FloatRangeSlider, IntSlider, Checkbox, Layout, Output, Label
from ipywidgets import Output, HBox, VBox, Checkbox, Layout, Label, IntRangeSlider
from matplotlib.lines import Line2D

import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\34699\OneDrive\Escritorio\ProyectoFinalSubastas\SubastasArteClasico\Datos\datos_subastas_procesado.csv')  
    Artistas = pd.read_csv(r'C:\Users\34699\OneDrive\Escritorio\ProyectoFinalSubastas\SubastasArteClasico\Datos\Artistas.csv')  
    return df, Artistas

df, Artistas = load_data()


# Título de la aplicación
st.title("Oferta de pintura clásica en la casa de subastas Christie's")

pagina = st.selectbox("Selecciona una página:", ["Distribución de la oferta por categorías", "Distribución por años", "Análisis de precios"])



if pagina == "Distribución de la oferta por categorías":
    
    
    min_year = int(df['Año de nacimiento'].min())
    max_year = int(df['Año de nacimiento'].max())
    
    # Función para representar obras y artistas por países
    year_range_slider = st.slider(
    'Selecciona el rango de años',
    min_year,
    max_year,
    (min_year, max_year)
)


    # Filtrar advertencias específicas
    warnings.filterwarnings("ignore")

    graf_output = Output()



    def representar_obras_y_artistas_por_paises(intervalo_años):
        
        Años = range(intervalo_años[0], intervalo_años[1] , 1) 
        
        Paises = list(Artistas['País'].unique())
        Paises.remove('Desconocido')# Intervalo de años basado en el slider
        
        idxFilasArt = [list(Artistas.loc[(Artistas['País'] == pais) & 
                                        (Artistas['Año de nacimiento'].isin(Años))].index) for pais in Paises]
        idxFilasOb = [list(df.loc[(df['País'] == pais) & 
                                (df['Año de nacimiento'].isin(Años))].index) for pais in Paises]

        graf_paisesob =[[ pais, len(idxFilasOb[i]), sum( df.loc[ j , 'Vendido'] for j in idxFilasOb[i]) ] for i, pais in enumerate(Paises)]
        graf_paisesart =[[ pais , len(idxFilasArt[i]), sum((1 if Artistas.loc[j,'Apariciones'] > 1 else 0) for j in idxFilasArt[i])]  for i, pais in enumerate(Paises)] 

        graf_paisesob = sorted(graf_paisesob, key=lambda x: x[1], reverse=True)
        graf_paisesart = sorted(graf_paisesart, key=lambda x: x[1], reverse=True)

        graf_paisesob = graf_paisesob [:10]
        graf_paisesart = graf_paisesart [:10]

        obras_y = [pais[0] for i, pais in enumerate(graf_paisesob)]
        obras_x = [pais[1] for i, pais in enumerate(graf_paisesob)]
        ventas_x = [pais[2] for i, pais in enumerate(graf_paisesob)]

        artistas_y = [pais[0] for i, pais in enumerate(graf_paisesart)]
        artistas_x = [pais[1] for i, pais in enumerate(graf_paisesart)]
        conocido_x = [pais[2] for i, pais in enumerate(graf_paisesart)]
        
        with graf_output:
            
            graf_output.clear_output(wait=True)
            
            fig = plt.figure(figsize=(40, 30))
            gs = fig.add_gridspec(9, 8)

            ax1 = fig.add_subplot(gs[:4, :5])
            ax2 = fig.add_subplot(gs[5:, :5])
            ax3 = fig.add_subplot(gs[0, 5:])
            ax4 = fig.add_subplot(gs[1:5, 5:])

            ax5 = fig.add_subplot(gs[6:,5:])


            sns.set_color_codes("pastel")
            sns.barplot(y=obras_y, x=obras_x, ax=ax1, color='grey', alpha=1, edgecolor='black', label='Obras por país')  # x e y pasan aquí correctamente
            sns.barplot(y=obras_y, x=ventas_x, ax=ax1, color='brown',alpha=1, edgecolor='black', label='Ventas por país')
            # Configurar el gráfico
            ax1.set_title("Número de obras por país", fontsize=50, fontweight='bold', loc='left', pad=30)
            ax1.set_xlim(0, max(obras_x)*1.05)

            ax1.set_xticks([0 ,np.array(obras_x).mean(), max(obras_x)])
            
            labels = ax1.get_yticklabels() 
            for label in labels:  
                label.set_fontsize(30)
            ax1.set_yticklabels(labels)
            
            for i in range(3):  
                labels[i].set_fontweight('bold') 
                labels[i].set_fontsize(30)
                
            ax1.set_yticklabels(labels)
            
            for label in ax1.get_xticklabels():  
                label.set_fontsize(30) 
            
            sns.despine(right=True, top=True)
            

            ax1.axvline(np.mean(obras_x), linestyle='--', color='black', linewidth=1.2)  # Media de obras
            ax1.axvline(np.quantile(obras_x, 0.25), linestyle='--', color='black', linewidth=0.5) 
            ax1.axvline(np.quantile(obras_x, 0.75), linestyle='--', color='black', linewidth=0.5) 

            ax1.legend(fontsize=30)


            # Plotear la cantidad total de artistas por país
            sns.set_color_codes("pastel")
            sns.barplot(y=artistas_y, x=artistas_x, ax=ax2, color='orange', edgecolor='black', label='Nº de artistas')  # x e y pasan aquí correctamente
            sns.barplot(y=artistas_y, x=conocido_x, ax=ax2, color='brown', edgecolor='black', label='Artistas con más de una obra') 
            # Configurar el gráfico
            ax2.set_title("Número de artistas por país", fontsize=50, fontweight='bold', loc='left', pad=30)
            ax2.set_xlim(0, max(artistas_x)*1.05)
            
            labels = ax2.get_yticklabels()  # Obtener etiquetas actuales
            for label in labels:  
                label.set_fontsize(30)
            ax2.set_yticklabels(labels)
            
            for i in range(3):  # Aplicar formato a las tres primeras etiquetas
                labels[i].set_fontweight('bold') 
                labels[i].set_fontsize(30)
            ax2.set_yticklabels(labels)
            
            for label in ax2.get_xticklabels():  
                label.set_fontsize(30) 
            
            sns.despine(right=True, top=True)
            
            ax2.axvline(np.mean(artistas_x), linestyle='--', color='black', linewidth=1.2)

            ax2.set_xticks([0,np.array(conocido_x).mean(), np.array(artistas_x).mean(), max(artistas_x)])
            ax2.legend(fontsize=30)


            idxFilasObperiodo = list(df.loc[(df['Año de nacimiento'].isin(Años)) & (df['Vendido'] == 1)].index)

            # Encuentra la obra más cara vendida
            if not idxFilasObperiodo:
                print("No hay obras vendidas en el intervalo seleccionado.")
                    

            obra_index = max(idxFilasObperiodo, key=lambda x: df.loc[x, 'Precio_num'])  # Obtener el índice de la obra más cara
            obra = df.iloc[obra_index] 



            ax3.set_title("Pintura más cara de este periodo:", fontsize=40, fontweight='bold', loc='left')

            # Eliminar ejes
            ax3.axis('off')  # Esto quita todos los ejes para que no haya recuadro

            # Texto sobre la obra
            ax3.text(0, 0.7, f"{obra['Nombre']}, {int(obra['Año de nacimiento'])}, {obra['País']}. {obra['Género']}.",
                    fontsize=30, ha='left', va='top', color='black', transform=ax3.transAxes)
            ax3.text(0, 0.45, f"{obra['Medio']} sobre {obra['Soporte']} de {round(obra['Alto'],2)} x {round(obra['Ancho'],2)} cm.",
                    fontsize=30, ha='left', va='top', color='black', transform=ax3.transAxes)
            ax3.text(0, 0.1, f"Vendida en {obra['Ciudad']} el {obra['Fecha']} por {int(obra['Precio_num'])} €.",
                    fontsize=30, ha='left', va='top', color='black', transform=ax3.transAxes)

            response = requests.get(obra['URL imagen'])
            if response.status_code == 200:  # Verificar si la solicitud tuvo éxito
                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                ax4.imshow(image)
                ax4.axis('off')  # Quitar el eje
            else:
                print("Error al cargar la imagen.")

    

            Generos = list(df['Género'].unique())

            idxFilasOb = [list(df.loc[(df['Género'] == genero) & 
                                    (df['Año de nacimiento'].isin(Años))].index) for genero in Generos]

            graf_generos =[[genero, len(idxFilasOb[i]), sum([df.loc[j,'Vendido'] for j in idxFilasOb[i]]), sum([(df.loc[j,'Precio_num'] if pd.notnull(df.loc[j,'Precio_num']) else 0) for j in idxFilasOb[i]])/sum([df.loc[j,'Vendido'] for j in idxFilasOb[i]]) ] for i, genero in enumerate(Generos)]


            for i, genero in enumerate(graf_generos):
                    if genero[1] <= 30:
                            graf_generos.remove(genero)


            graf_generos = sorted(graf_generos, key=lambda x: x[3], reverse=True)
            preciomedio = np.array([genero[3] for genero in graf_generos]).mean()  
            for genero in graf_generos:
                    genero[1] = genero[1] - genero[2]
                    genero[3] = genero[3]/preciomedio

            sectores=[]
            for  i,genero in enumerate(graf_generos):
            
                    sectores.append([genero[0], genero[2], genero[3]])
                    sectores.append([" ", genero[1], 1])
            

            etiquetas = [sector[0] for i,sector in enumerate(sectores)]
            valores = [sector[1] for i,sector in enumerate(sectores)]                                              
            radios = [sector[2] for i,sector in enumerate(sectores)] 

            valortotal = sum(valores)

            max_radio=max(radios)
            radios=[(radio/(max_radio)  if i % 2 == 0 else 1) for i, radio in enumerate(radios)]


            ax5.set_title("Obras según el género", fontsize=50, fontweight='bold', loc='left', pad=20)

            cmap = plt.colormaps["tab20c"]
            inner_colors = cmap(range(len(etiquetas)))  




            for i, genero in enumerate(graf_generos):
            
                    try:
                            vals =np.array([np.array(valores[:2*i]).sum(), valores[2*i], np.array(valores[2*i+1:]).sum()])
                            colores=[(1, 1, 1, 0), inner_colors[2*i], (1, 1, 1, 0)]
                    except:
                            break
                    
                    ax5.pie(vals, radius=0.7, colors=colores,
                            
                    wedgeprops=dict(width=0.35,edgecolor=None, linewidth=0))
            
            for i, genero in enumerate(graf_generos):
            
                    try:
                            vals =np.array([np.array(valores[:2*i +1]).sum(), valores[2*i +1], np.array(valores[2*i+2:]).sum()])
                            colores=[(1, 1, 1, 0), 'orange', (1, 1, 1, 0)]
                    except:
                            break
                    
                    ax5.pie(vals, radius=0.7, colors=colores,
                            
                    wedgeprops=dict(width=0.35,edgecolor=None, linewidth=0))
            
            for i, genero in enumerate(graf_generos):
            
                    try:
                            vals =np.array([np.array(valores[:2*i]).sum(), valores[2*i], np.array(valores[2*i+1:]).sum()])
                            etiq=[None, etiquetas[2*i], None]
                            colores=[(1, 1, 1, 0), inner_colors[2*i + 1], (1, 1, 1, 0)]
                    except:
                            break
                    
                    ax5.pie(vals, radius=0.7 + radios[2*i]/2, colors=colores,
                            
                    wedgeprops=dict(width=radios[2*i]/2 ,edgecolor=None, linewidth=0), labels=etiq, textprops=dict(fontsize=25))
            
            
            acum_angle = 0
            for i in range(len(graf_generos)):

                    ax5.plot(
                            [0.35 * np.cos(acum_angle),(0.7 + radios[2*i]/2 )* np.cos(acum_angle)],
                            [0.35 * np.sin(acum_angle), (0.7 + radios[2*i]/2 )* np.sin(acum_angle)],
                            color='black', linestyle='--', linewidth=0.7
                    )

                    # Sumar el ángulo al acumulado para la próxima iteración
                    acum_angle += (valores[2*i])/ valortotal * 2 * np.pi
                    
                    # Dibujar la línea radial usando el ángulo acumulado
                    ax5.plot(
                            [0.35 * np.cos(acum_angle), (0.7 + radios[2*i]/2 )* np.cos(acum_angle)],
                            [0.35 * np.sin(acum_angle), (0.7 + radios[2*i]/2 )* np.sin(acum_angle)],
                            color='black', linestyle='--', linewidth=0.7
                    )

                    # Sumar el ángulo al acumulado para la próxima iteración
                    acum_angle += (valores[2*i + 1])/ valortotal * 2 * np.pi 
            
            
            
            circle1 = Circle((0, 0), 0.7, color='black', fill=False, linewidth=0.8, linestyle='--')
            circle2 = Circle((0, 0), 0.35, color='black', fill=False, linewidth=0.8, linestyle='--')

            ax5.add_patch(circle1)
            ax5.add_patch(circle2)
            
            
            
            etiquetas_leyenda = ['Amplitud ~ Nº obras', 'Radio exterior ~ precio medio', 'No vendidas']

            leyenda_entries = [
            Line2D([0], [0], marker='*',color='none',  markerfacecolor='black'), 
            Line2D([0], [0], marker='*',color='none',  markerfacecolor='black') ,
            Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', markersize=10)  # No vendidas (naranja)
            ]

            # Añadir la leyenda personalizada al gráfico
            ax5.legend(leyenda_entries, etiquetas_leyenda, loc='center left', bbox_to_anchor=(1, -0.1), fontsize=30)



            plt.tight_layout(h_pad= 30, w_pad=20)
            plt.show()
        st.pyplot(fig)
        

    # Mostrar resultados
    representar_obras_y_artistas_por_paises(year_range_slider)



elif pagina == "Distribución por años":
    
    def representar_obras_y_precios_por_año(paiseselegidos, generoselegidos, intervalo_años):
        # Lógica para calcular datos
        graf_Años = pd.DataFrame(columns=['Año', 'TObras', 'TVentas', 'MinEst Medio', 'MaxEst Medio', 'PrecioMv', 'PrecioMs'])
        min_año = int(df['Año de nacimiento'].min())
        max_año = int(df['Año de nacimiento'].max()) 

        rango = (intervalo_años[1] - intervalo_años[0]) / 15
        rango = int(rango)

        Años = range(intervalo_años[0] + rango, intervalo_años[1] - rango + 1, (int(rango/8) if rango >= 8 else 1))
        idxFilas = [list(df.loc[(abs(df['Año de nacimiento'] - año) <= rango) & 
                                (df['País'].isin(paiseselegidos)) & 
                                (df['Género'].isin(generoselegidos))].index) for año in Años]
        
        idFilas = [list(df.loc[(df['Año de nacimiento'] == año) & 
                                (df['País'].isin(paiseselegidos)) & 
                                (df['Género'].isin(generoselegidos))].index) for año in range(intervalo_años[0], intervalo_años[1] + 1)]
        
        TObras = []
        TVentas = []   
        PrecioMv = [] 
        PrecioMs = []    
        MaxEst = []    
        MinEst = []
        
        graf_Años['Año'] = Años
        
        for i, año in enumerate(graf_Años['Año']):
            TObras.append((len(idxFilas[i]) if len(idxFilas[i]) > 0 else 1) / (2*rango))
            TVentas.append((sum(df.loc[j, 'Vendido'] for j in idxFilas[i]) if sum(df.loc[j, 'Vendido'] for j in idxFilas[i]) > 0 else 1) / (2*rango))
            
            PrecioMv.append(round(sum((df.loc[j, 'Precio_num'] if df.loc[j, 'Vendido'] == 1 else 0) for j in idxFilas[i]) / (2*rango* TVentas[i]), 2) if TVentas[i] > 0 else 0 )
            PrecioMs.append(round(sum([df.loc[j, 'Precio_num'] for j in idxFilas[i] if pd.notnull(df.loc[j, 'Precio_num'])]) / (2*rango* TObras[i]), 2) if TObras[i] > 0 else 0)        
            MaxEst.append(round(sum([df.loc[j, 'Máximo estimado'] for j in idxFilas[i] if pd.notnull(df.loc[j, 'Máximo estimado'])]) / (2*rango* TObras[i]), 2) if TObras[i] > 0 else 0)
            MinEst.append(round(sum([df.loc[j, 'Mínimo estimado'] for j in idxFilas[i] if pd.notnull(df.loc[j, 'Mínimo estimado'])]) / (2*rango* TObras[i]), 2) if TObras[i] > 0 else 0)
            
        graf_Años['TObras'] = TObras
        graf_Años['TVentas'] = TVentas
        graf_Años['PrecioMv'] = PrecioMv
        graf_Años['PrecioMs'] = PrecioMs
        graf_Años['MaxEst Medio'] = MaxEst
        graf_Años['MinEst Medio'] = MinEst

        TotalObras = sum(TObras)
        TotalVentas = sum(TVentas)
        TotalObrasABS = sum(len(idFilas[i]) for i, año in enumerate(idFilas))
        TotalVentasABS = sum(sum([df.loc[j, 'Vendido'] for j in idFilas[i]]) for i, año in enumerate(idFilas))
        
        Porcentaje = round((TotalVentasABS/TotalObrasABS) * 100, 2) if TotalObrasABS > 0 else 0


        PrecioMvGlobal = (sum(PrecioMv[i] * TVentas[i] for i in range(len(PrecioMv))) / TotalVentas) if TotalVentas > 0 else 0
        PrecioMsGlobal = (sum(PrecioMs[i] * TObras[i] for i in range(len(PrecioMs))) / TotalObras) if TotalObras > 0 else 0
        EstMaxMGlobal = (sum(MaxEst[i] * TObras[i] for i in range(len(MaxEst))) / TotalObras) if TotalObras > 0 else 0
        EstMinMGlobal = (sum(MinEst[i] * TObras[i] for i in range(len(MinEst))) / TotalObras) if TotalObras > 0 else 0

        # Visualización
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9.5, 9))
        sns.set_theme(style="white")

        # Primer gráfico
        axes[0].set_title("Número de obras", fontsize=20, fontweight='bold', loc='left', pad=30)
        sns.lineplot(x="Año", y="TObras", data=graf_Años, label='Subastadas', color='blue', ax=axes[0])
        sns.lineplot(x="Año", y="TVentas", data=graf_Años, label='Vendidas', color='red', ax=axes[0])

        # Rellenar área entre líneas
        axes[0].fill_between(graf_Años['Año'], graf_Años['TVentas'], color='red', alpha=0.15)
        axes[0].fill_between(graf_Años['Año'], graf_Años['TObras'], graf_Años['TVentas'], color='blue', alpha=0.3)

        # Configuración del gráfico
        axes[0].set_xlabel("")
        axes[0].set_ylabel("")
        yticks = axes[0].get_yticks()
        axes[0].set_yticks([yticks[1], yticks[int(len(yticks) / 2)], yticks[-2]])
        axes[0].grid(False)
        axes[0].spines['bottom'].set_color('black')
        axes[0].spines['left'].set_color('black')

        # Segundo gráfico
        axes[1].set_title("Precios medios y valores estimados", fontsize=20, fontweight='bold', loc='left', pad=30)
        sns.lineplot(x="Año", y="MaxEst Medio", data=graf_Años, label='Máx. estimado', color='orange', ax=axes[1])
        sns.lineplot(x="Año", y="PrecioMv", data=graf_Años, label='Precio Medio por venta', color='red', ax=axes[1])
        sns.lineplot(x="Año", y="PrecioMs", data=graf_Años, label='Medio por subasta', color='blue', ax=axes[1], linewidth=0.7, linestyle='--')
        sns.lineplot(x="Año", y="MinEst Medio", data=graf_Años, label='Mín. estimado', color='orange', ax=axes[1])

        axes[1].plot([min_año - 10, max_año + 10], [graf_Años['PrecioMv'].quantile(0.25), graf_Años['PrecioMv'].quantile(0.25)], color='brown', linewidth=0.5, linestyle='--')
        axes[1].plot([min_año - 10, max_año + 10], [PrecioMvGlobal, PrecioMvGlobal], color='red', linewidth=0.8)
        axes[1].plot([min_año - 10, max_año + 10], [graf_Años['PrecioMv'].quantile(0.75), graf_Años['PrecioMv'].quantile(0.75)], color='brown', linewidth=0.5, linestyle='--')

        axes[1].fill_between(graf_Años['Año'], graf_Años['MaxEst Medio'], graf_Años['MinEst Medio'], color='orange', alpha=0.4)

        axes[1].fill_between([min_año+30, max_año-29], [graf_Años['PrecioMv'].quantile(0.25), graf_Años['PrecioMv'].quantile(0.25)], 
                            [graf_Años['PrecioMv'].quantile(0.75), graf_Años['PrecioMv'].quantile(0.75)], color='limegreen', alpha=0.3)
        axes[1].fill_between(graf_Años['Año'], graf_Años['PrecioMv'], graf_Años['PrecioMs'], color='blue', alpha=0.3)

        axes[1].set_xlim(intervalo_años[0], intervalo_años[1])
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")
        axes[1].set_yticks([graf_Años['PrecioMv'].quantile(0.25), PrecioMvGlobal, graf_Años['PrecioMv'].quantile(0.75), graf_Años['PrecioMv'].max()])
        axes[1].grid(False)
        axes[1].spines['bottom'].set_color('black')
        axes[1].spines['left'].set_color('black')

        plt.tight_layout(h_pad=4)
        st.pyplot(fig)

        # Visualización de Totales
        st.subheader("Totales")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total de obras subastadas", TotalObrasABS)
            st.metric("Total de obras vendidas", TotalVentasABS)
            st.metric("Porcentaje de ventas", f"{Porcentaje} %")

        with col2:
            st.metric("Precio medio por venta", f"${PrecioMvGlobal:.2f}")
            st.metric("Precio medio por subasta", f"${PrecioMsGlobal:.2f}")
            st.metric("Estimado máximo medio", f"${EstMaxMGlobal:.2f}")
            st.metric("Estimado mínimo medio", f"${EstMinMGlobal:.2f}")

    # Crear la interfaz de Streamlit    
    st.title('Análisis de Obras de Arte')

    # Lista de países y géneros
    Paises = ['España', 'Estados Unidos', 'Alemania', 'Bélgica', 'Paises Bajos', 'Francia', 'Reino Unido', 'Italia', 'Austria', 'Suiza', 'Desconocido']
    Generos = df['Género'].unique()

    # Widgets de selección usando Streamlit
    paiseselegidos = st.multiselect('Selecciona los países', Paises, default=Paises)
    generoselegidos = st.multiselect('Selecciona los géneros', Generos, default=Generos)

    # Slider de selección de años
    min_year = int(df['Año de nacimiento'].min())
    max_year = int(df['Año de nacimiento'].max())
    intervalo_años = st.slider('Periodo de años', min_year, max_year, (min_year, max_year))

    # Llamar a la función para representar los gráficos
    representar_obras_y_precios_por_año(paiseselegidos, generoselegidos, intervalo_años)

elif pagina == "Análisis de precios":
    
    Q1=df[df['Vendido']==1]['Precio_num'].quantile(0.25)
    Q3=df[df['Vendido']==1]['Precio_num'].quantile(0.75)
    QM=df[df['Vendido']==1]['Precio_num'].mean()
    Qm=df[df['Vendido']==1]['Precio_num'].median()
    IQR=Q3-Q1
    cinf=Q1 - 1.5*IQR
    csup=Q3 + 1.5*IQR

    def plot_first_graph(df):
        precios=[]

        for i,obra in df.iterrows():
            if obra['Vendido'] == 1:
                precios.append(obra['Precio_num'])

        precios=sorted(precios) #almacenamos los precios en una lista ordenada

        Total=len(precios) 

        min_pre = int(precios[0]) 
        max_pre = int(precios[-1])

        x_precios = []
        y_precios = []

        porK = 30

        paso = min_pre * porK

        x = 0

        while precios:
            
            precio=precios[0] 
            num=0
            
            while precio <= x + paso:
                
                num += 1
                try:
                    precio=precios.pop(0)
                except:
                    break

            y_precios.append(num)
            x_precios.append(x)
            
            x += paso
        y_de_trabajo = y_precios

            
        rep=1
        maxrep=10

        coefs=[]
        for L in range(1, maxrep + 2):
            coefs.append([(sum( 1/ ( (n + k)**2 ) for k in range( 0, L - n + 1 )) / sum( 1/j for j in range(1, L + 1))) for n in range(1, L + 1) ])

        while rep <= maxrep:
            
            L = rep
            y_val = []
        
            for i, x in enumerate(x_precios[:2000]):
                
                valsD = []
                valsI = []
                
                for n in range(1, L+1):
                    
                    try:
                        valsD.append(y_de_trabajo[i+n])
                    except:
                        pass
                    
                    try:
                        valsI.append(y_de_trabajo[i-n])
                    except:
                        pass
                    
                LD = len(valsD)
                LI = len(valsI)
                vals = []
                
                for j,val in enumerate(valsI):
                    vals.append(val*coefs[LI-1][j] * (LD/(LD + LI)))
                    
                for j,val in enumerate(valsD):
                    vals.append(val*coefs[LD-1][j] * (LI/(LD + LI)))
                    
                y_val.append(np.array(vals).sum())
                
                if i > rep:
                    idf = rep
                    ultimos = y_val[-idf :]
                    y_val = y_val[: - idf]
                    I_f = sum(ultimos)/paso
                    I_0 = sum(y_precios[i - idf + 1: i + 1])/paso
                    trozo = (I_0 - I_f)/idf
                    for k, ult in enumerate(ultimos):
                        nuevo = ult + 2*(idf-k)*trozo
                        y_val.append(nuevo)
                

            y_de_trabajo =list(np.array(y_de_trabajo[:2000])*((rep-1)/rep) + np.array( y_val)*(1/rep))
            
            
            rep += 1


        plt.figure(figsize=(15, 5))

        plt.fill_between([Q1, Q3], [max(y_precios)*1.1,max(y_precios)*1.1], color='lightblue', alpha=0.6)

        for i in range(len(x_precios)):
            try:
                plt.plot([x_precios[:int(4000/porK)][i], x_precios[:int(4000/porK)][i+1]], [y_precios[:int(4000/porK)][i], y_precios[:int(4000/porK)][i]], color='red', linewidth=0.8)
                plt.fill_between([x_precios[:int(4000/porK)][i], x_precios[:int(4000/porK)][i+1]], [y_precios[:int(4000/porK)][i], y_precios[:int(4000/porK)][i]], color='orange', alpha=0.8)
            except:
                break

        sns.lineplot(x=x_precios[:int(4000/porK)], y=y_val[:int(4000/porK)], color='brown')

        plt.axvline(Q1, linestyle='--', color='grey', linewidth=1)  
        plt.axvline(Q3, linestyle='--', color='grey', linewidth=1) 
        plt.axvline(QM, linestyle='--', color='red', linewidth=1.7) 
        plt.axvline(Qm, linestyle='--', color='blue', linewidth=1)


        plt.fill_between([csup, x_precios[int(4000/porK)]], [max(y_precios)*1.1,max(y_precios)*1.1], color='grey', alpha=0.7)

        plt.axvline(csup, linestyle='--', color='black', linewidth=0.7) 

        # Configuración adicional de la gráfica
        plt.title('Distribución de la variable precio de venta')
        st.pyplot(plt)

    def plot_second_graph(df):
        precios=[]

        for i,obra in df.iterrows():
            if obra['Vendido'] == 1:
                precios.append(obra['Precio_num'])

        precios=sorted(precios) #almacenamos los precios en una lista ordenada

        Total=len(precios) 

        min_pre = int(precios[0]) 
        max_pre = int(precios[-1])

        x_tamaño = []
        y_tamaño = []

        tporK = 100

        tpaso = min_pre * tporK

        x = 0

        while precios:
            
            precio=precios[0] 
            num=0
            
            while precio <= x + tpaso:
                
                num += precio
                try:
                    precio=precios.pop(0)
                except:
                    break

            y_tamaño.append(num)
            x_tamaño.append(x)
            
            x += tpaso
        y_de_trabajo = y_tamaño

        rep=1
        maxrep=20

        coefs=[]
        for L in range(1, maxrep + 2):
            coefs.append([(sum( 1/ ( (n + k)**2 ) for k in range( 0, L - n + 1 )) / sum( 1/j for j in range(1, L + 1))) for n in range(1, L + 1) ])

        while rep <= maxrep:
            
            L = rep
            y_tval = []
        
            for i, x in enumerate(x_tamaño):
                
                valsD = []
                valsI = []
                
                for n in range(1, L+1):
                    
                    try:
                        valsD.append(y_de_trabajo[i+n])
                    except:
                        pass
                    
                    try:
                        valsI.append(y_de_trabajo[i-n])
                    except:
                        pass
                    
                LD = len(valsD)
                LI = len(valsI)
                vals = []
                
                for j,val in enumerate(valsI):
                    vals.append(val*coefs[LI-1][j] * (LD/(LD + LI)))
                    
                for j,val in enumerate(valsD):
                    vals.append(val*coefs[LD-1][j] * (LI/(LD + LI)))
                    
                y_tval.append((np.array(vals).sum()*(1/rep) + y_de_trabajo[i]*((rep-1)/rep)))
                
            y_de_trabajo = y_tval
            
            rep += 1

        plt.figure(figsize=(15, 5))

        for i in range(len(x_tamaño)):
            try:
                plt.plot([x_tamaño[:int(10000/tporK)][i], x_tamaño[:int(10000/tporK)][i+1]], [y_tamaño[:int(10000/tporK)][i], y_tamaño[:int(10000/tporK)][i]], color='red', linewidth=1.2)
                plt.fill_between([x_tamaño[:int(10000/tporK)][i], x_tamaño[:int(10000/tporK)][i+1]], [y_tamaño[:int(10000/tporK)][i], y_tamaño[:int(10000/tporK)][i]], color='red', alpha=0.6)
            except:
                break


        plt.axvline(Q1, linestyle='--', color='grey', linewidth=1)  
        plt.axvline(Q3, linestyle='--', color='grey', linewidth=1) 
        plt.axvline(Qm, linestyle='--', color='blue', linewidth=1)

        sns.lineplot(x=x_tamaño[:int(10000/tporK)], y=y_tval[:int(10000/tporK)], color='blue')


        plt.axvline(csup, linestyle='--', color='black', linewidth=0.7) 

        plt.title('Tamaño del mercado')
        st.pyplot(plt)

    def plot_third_graph(df):
        logprecios=[]

        for i,obra in df.iterrows():
            if df.loc[i,'Vendido'] == 1:
                df.loc[i,'logP'] = math.log(obra['Precio_num'])
                logprecios.append(math.log(obra['Precio_num']))


        logprecios=sorted(logprecios) 


        min_log_pre = int(logprecios[0]) 
        max_log_pre = int(logprecios[-1])

        x_logprecios = []
        y_logprecios = []


        lpaso = min_log_pre/ 17

        x = min_log_pre - 2*lpaso

        while logprecios:
            
            logprecio=logprecios[0] 
            num=0
            
            while logprecio <= x + lpaso:
                
                num += 1
                try:
                    logprecio=logprecios.pop(0)
                except:
                    break

            y_logprecios.append(num)
            x_logprecios.append(x)
            
            x += lpaso
            
        y_de_trabajo = y_logprecios

        rep=1
        maxrep=6

        coefs=[]
        for L in range(1, maxrep + 2):
            coefs.append([(sum( 1/ ( (n + k)**2 ) for k in range( 0, L - n + 1 )) / sum( 1/j for j in range(1, L + 1))) for n in range(1, L + 1) ])

        while rep <= maxrep:
            
            L = rep
            y_logval = []
        
            for i, x in enumerate(x_logprecios):
                
                valsD = []
                valsI = []
                
                for n in range(1, L+1):
                    
                    try:
                        valsD.append(y_de_trabajo[i+n])
                    except:
                        pass
                    
                    try:
                        valsI.append(y_de_trabajo[i-n])
                    except:
                        pass
                    
                LD = len(valsD)
                LI = len(valsI)
                vals = []
                
                for j,val in enumerate(valsI):
                    vals.append(val*coefs[LI-1][j] * (LD/(LD + LI)))
                    
                for j,val in enumerate(valsD):
                    vals.append(val*coefs[LD-1][j] * (LI/(LD + LI)))
                    
                y_logval.append((np.array(vals).sum()*(1/rep) + y_de_trabajo[i]*((rep-1)/rep)))
                
            y_de_trabajo = y_logval
            
            rep += 1

        Qlog1=df[df['Vendido']==1]['logP'].quantile(0.25)
        Qlog3=df[df['Vendido']==1]['logP'].quantile(0.75)
        QlogM=df[df['Vendido']==1]['logP'].mean()
        IQRlog=Qlog3-Qlog1
        cloginf=Qlog1 - 1.5*IQRlog
        clogsup=Qlog3 + 1.5*IQRlog
        plt.figure(figsize=(15, 5))

        plt.fill_between([Qlog1, Qlog3], [max(y_logprecios)*1.1,max(y_logprecios)*1.1], color='lightblue', alpha=0.6)

        for i,x in enumerate(x_logprecios):
            
            try:
                plt.plot([x_logprecios[i], x_logprecios[i+1]], [y_logprecios[i], y_logprecios[i]], color='red', linewidth=0.8)
                plt.fill_between([x_logprecios[i], x_logprecios[i+1]], [y_logprecios[i], y_logprecios[i]], color='orange', alpha=0.8)
            except:
                break


        sns.lineplot(x=[equis + lpaso for equis in x_logprecios], y=y_logval, color='brown')


        plt.axvline(Qlog1, linestyle='--', color='grey', linewidth=1)  
        plt.axvline(Qlog3, linestyle='--', color='grey', linewidth=1) 
        plt.axvline(QlogM, linestyle='--', color='red', linewidth=1.7) 

        plt.fill_between([min_log_pre - 2*lpaso,cloginf], [max(y_logprecios)*1.1,max(y_logprecios)*1.1], color='grey', alpha=0.7)
        plt.fill_between([clogsup, max_log_pre + lpaso], [max(y_logprecios)*1.1,max(y_logprecios)*1.1], color='grey', alpha=0.7)



        plt.axvline(cloginf, linestyle='--', color='black', linewidth=0.7) 
        plt.axvline(clogsup, linestyle='--', color='black', linewidth=0.7) 

        # Configuración adicional de la gráfica
        plt.title('Distribución del logaritmo del precio')

        st.pyplot(plt)

    def main():
        st.title("Análisis de Precios ")

        st.subheader("Primer gráfico: Distribución de precios de venta")
        plot_first_graph(df)

        st.subheader("Segundo gráfico: Tamaño del mercado")
        plot_second_graph(df)

        st.subheader("Tercer gráfico: Distribución del logaritmo del precio")
        plot_third_graph(df)

    if __name__ == "__main__":
        main()


        
