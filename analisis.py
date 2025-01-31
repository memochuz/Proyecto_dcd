import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import re
import datetime
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import silhouette_score
from app_funciones import *
import matplotlib.ticker as ticker  # Importar formateador de números


##########################################################
##########################################################
##########################################################


def analisis_page():
    st.header("Página: Análisis")

    ##########################################################

    st.markdown(
        """
        Acontinuación trataremos los siguientes puntos para poder complementar lo que encontremos con nuestros modelos:
        1. Total por mes.
        2. Total por mes de los 3 mejores productos.
        3. Total por mes de los siguientes 4 mejores productos
        4. Las 10 transacciones que generan la mayor cantidad de dinero.
        5. Los 10 productos que generan la mayor cantidad de dinero.
        6. Los 10 productos que generan la menor cantidad de dinero.
        7. Los 10 productos más caros .
        8. Los 10 producto más baratos.
        9. Los 10 países que más gastaron.
        """
        )
    
    # st.write(st.session_state.datos.columns)

    # Create New Columns : Month
    st.session_state.datos['Month'] = pd.DatetimeIndex(st.session_state.datos['Date']).month

    
    ##########################################################
    ##########################################################
    ################## TABLAS Y  GRAFICA #####################
    ##########################################################
    ##########################################################

    # GRÁFICAS Y TABLAS

    total_spent_max = st.session_state.datos.groupby(["TransactionNo"])["TotalSpent"].sum()
    total_spent_max = total_spent_max.reset_index()

    # TABLA
    total_spent_max = total_spent_max.sort_values(by="TotalSpent", ascending=False).head(10)

    # GRAFICA
    # Convertir índice en columna para que sea un DataFrame
    total_spent_max_graph = bars(total_spent_max,
         "Top 10 transacciones con mayor gasto",
         "Número de transacción",
         "Total"
         )

    
    ##########################################################

    products_max = st.session_state.datos.groupby(["ProductName"])["TotalSpent"].sum()
    products_max = products_max.reset_index()

    # TABLA
    products_max = products_max.sort_values(by = "TotalSpent", ascending=False).head(10)
    # Convertir índice en columna para que sea un DataFrame
    
    # GRAFICA
    products_max_graph = bars(products_max,
         "Los 10 productos que generan más dinero",
         "Producto",
         "Total"
         )
   
    ##########################################################

    products_min = st.session_state.datos.groupby(["ProductName"])["TotalSpent"].sum()
    products_min = products_min.reset_index()

    # TABLA
    products_min = products_min.sort_values(by = "TotalSpent", ascending=True).head(10)
    # Convertir índice en columna para que sea un DataFrame
    
    # GRAFICA
    products_min_graph = bars(products_min,
         "Los 10 productos que generan menos dinero",
         "Producto",
         "Total"
         )
    
    ##########################################################

    productos_caros = st.session_state.datos.groupby("ProductName")["Price"].max()
    productos_caros = productos_caros.reset_index()

    # TABLA
    productos_caros = productos_caros.sort_values(by="Price", ascending=False).head(10) 

    # GRAFICA
    productos_caros_graph = bars(productos_caros,
         "Los 10 productos más caros",
         "Producto",
         "Precio"
         )
           
    ##########################################################

    productos_baratos = st.session_state.datos.groupby("ProductName")["Price"].max()
    productos_baratos = productos_baratos.reset_index()

    # TABLA
    productos_baratos = productos_baratos.sort_values(by="Price", ascending=True).head(10) 

    # GRAFICA
    productos_baratos_graph = bars(productos_baratos,
         "Los 10 productos más baratos",
         "Producto",
         "Precio"
         )
    
    ##########################################################

    paises_max = st.session_state.datos.groupby("Country")["TotalSpent"].sum()
    paises_max = paises_max.reset_index()

    # TABLA
    paises_max = paises_max.sort_values(by = "TotalSpent", ascending=False).head(10) 

    # GRAFICA
    paises_max_graph = bars(paises_max,
         "Los 10 países que más gastaron",
         "Producto",
         "Total"
         )
    

    ##########################################################

    meses_total = st.session_state.datos.groupby("Month")["TotalSpent"].sum()

    # TABLA
    meses_total = meses_total.reset_index()

    plt.style.use("dark_background")  
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(meses_total.iloc[:,0], meses_total.iloc[:,1], marker='o', linestyle='-')
    # Personalizar etiquetas y títulos
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Mes", fontsize=12)
    ax.set_ylabel("Total", fontsize=12)
    ax.set_title("Total por mes", fontsize=14)
    ax.tick_params(axis='x') 
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    


    ##########################################################


    # LISTA QUE CONTIENE DATAFRAMES, QUE CONTIENEN EL GASTO POR MES DE LOS 7 PRODUCTOS MAS VENDIDOS

    productos_mes_top = st.session_state.datos.groupby(["ProductName","Month"])["TotalSpent"].sum()
    productos_mes_top = productos_mes_top.reset_index()
    productos_mes_top_prod_name = list(products_max["ProductName"])
    mes_top_df_1 = []
    mes_top_df_2= []

    # for product in set(products_max["ProductName"].unique()):
    for i ,product in enumerate(productos_mes_top_prod_name):
        if i <= 2:
            mes_top_df_1.append(productos_mes_top[productos_mes_top["ProductName"]== product])
        if i >= 3 and i<= 6:
            mes_top_df_2.append(productos_mes_top[productos_mes_top["ProductName"]== product])
    
    # for df in mes_top_df_1:
    #     st.write(df.iloc[0,0])
    # for df in mes_top_df_2:
    #     st.write(df.iloc[0,0])

 
    ##########################################################
    ##########################################################
    ############### VISUALIZAMOS GRAFICAS ####################
    ##########################################################
    ##########################################################

    # MES TOTAL
    plt.style.use("dark_background")  
    fig_mes_tot, ax_mes_tot = plt.subplots(figsize=(10, 5))
    ax_mes_tot.plot(meses_total.iloc[:,0], meses_total.iloc[:,1], marker='o', linestyle='-')
    # Agregar etiquetas con el valor de TotalSpent en cada punto
    for i, total in enumerate(meses_total.iloc[:, 1]):
        ax_mes_tot.text(meses_total.iloc[i, 0], total+100000, f"{total:,.0f}", fontsize=10, ha='center', va='bottom', color='white')
    ax_mes_tot.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, color="gray")
    for spine in ax_mes_tot.spines.values():
        spine.set_color("gray")  # Color gris
        spine.set_linewidth(0.7)  # Mismo grosor que el grid
        spine.set_alpha(0.7)  # Misma opacidad que el grid
    # Personalizar etiquetas y títulos
    ax_mes_tot.set_xticks(range(1, 13))
    ax_mes_tot.set_xlabel("Mes", fontsize=12)
    ax_mes_tot.set_ylabel("Total", fontsize=12)
    ax_mes_tot.set_title("Total por mes", fontsize=14)
    ax_mes_tot.tick_params(axis='x') 
    ax_mes_tot.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    col1,col2 = st.columns([1,1])
    with col1:
        st.pyplot(fig_mes_tot)
    with col2:
        st.dataframe(meses_total)

    st.markdown("<hr>", unsafe_allow_html=True)

    ##########################################################
    # GRAFICAR CADA PRODUCTO TOP 3
    fig_mes_top_3, ax_mes_top_3 = plt.subplots(figsize=(10, 5))

    for df in mes_top_df_1:
        product_name = df["ProductName"].iloc[0]  # Obtener nombre del producto
        ax_mes_top_3.plot(df["Month"], df["TotalSpent"], marker='o', linestyle='-', label=product_name)  # Línea con marcador

        # Etiqueta con el nombre del producto al final de la línea
        ax_mes_top_3.text(df["Month"].max()+0.2, df["TotalSpent"].iloc[-1], product_name, fontsize=8, verticalalignment='bottom')

    # Personalizar gráfica
    ax_mes_top_3.set_xticks(range(1, 13))  # Asegurar que los meses del 1 al 12 aparezcan
    ax_mes_top_3.set_xlabel("Mes", fontsize=12)
    ax_mes_top_3.set_ylabel("Total Gastado", fontsize=12)
    ax_mes_top_3.set_title("Gasto Mensual productos top 3", fontsize=14)
    ax_mes_top_3.tick_params(axis='x')
    ax_mes_top_3.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 0.8))  # Muestra las etiquetas de cada línea
    ax_mes_top_3.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, color="gray")
    for spine in ax_mes_top_3.spines.values():
        spine.set_color("gray")  # Color gris
        spine.set_linewidth(0.7)  # Mismo grosor que el grid
        spine.set_alpha(0.7)  # Misma opacidad que el grid




    col1,col2 = st.columns([1,1])
    with col1:
        st.pyplot(fig_mes_top_3)
    with col2:
        subcol1,subcol2,subcol3 = st.columns([1,1,1])
        with subcol1:
            st.dataframe(mes_top_df_1[0])
        with subcol2:
            st.dataframe(mes_top_df_1[1])
        with subcol3:
            st.dataframe(mes_top_df_1[2])
        
    st.markdown("<hr>", unsafe_allow_html=True)

    ##########################################################
    # GRAFICAR CADA PRODUCTO TOP QUITANDO TOP 3
    fig_mes_top, ax_mes_top = plt.subplots(figsize=(10, 5))

    for df in mes_top_df_2:
        product_name = df["ProductName"].iloc[0]  # Obtener nombre del producto
        ax_mes_top.plot(df["Month"], df["TotalSpent"], marker='o', linestyle='-', label=product_name)  # Línea con marcador

        # Etiqueta con el nombre del producto al final de la línea
        ax_mes_top.text(df["Month"].max() + 0.2, df["TotalSpent"].iloc[-1], product_name, fontsize=8, verticalalignment='bottom')

    # Personalizar gráfica
    ax_mes_top.set_xticks(range(1, 13))  # Asegurar que los meses del 1 al 12 aparezcan
    ax_mes_top.set_xlabel("Mes", fontsize=12)
    ax_mes_top.set_ylabel("Total Gastado", fontsize=12)
    ax_mes_top.set_title("Gasto Mensual Productos Top", fontsize=14)
    ax_mes_top.tick_params(axis='x')
    ax_mes_top.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))  # Muestra las etiquetas de cada línea
    ax_mes_top.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, color="gray")
    for spine in ax_mes_top.spines.values():
        spine.set_color("gray")  # Color gris
        spine.set_linewidth(0.7)  # Mismo grosor que el grid
        spine.set_alpha(0.7)  # Misma opacidad que el grid


    col1,col2 = st.columns([1,1])
    with col1:
        st.pyplot(fig_mes_top)
    with col2:
        subcol1,subcol2,subcol3,subcol4 = st.columns([1,1,1,1])
        with subcol1:
            st.dataframe(mes_top_df_2[0])
        with subcol2:
            st.dataframe(mes_top_df_2[1])
        with subcol3:
            st.dataframe(mes_top_df_2[2])
        with subcol4:
            st.dataframe(mes_top_df_2[3])

    # st.markdown("<hr>", unsafe_allow_html=True)
    ##########################################################
    ##########################################################
    ##########################################################
    
    # LISTAS CON LAS GRÁFICAS Y TABLAS

    tablas = [
        total_spent_max,
        products_max,
        products_min,
        productos_caros,
        productos_baratos,
        paises_max,
        # meses_total
    ]

    ##########################################################

    graficas = [
        total_spent_max_graph,
        products_max_graph,
        products_min_graph,
        productos_caros_graph,
        productos_baratos_graph,
        paises_max_graph,
        # fig
    ]

    for i in range(6):
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            st.pyplot(graficas[i])
        with col2:
            st.dataframe(tablas[i])