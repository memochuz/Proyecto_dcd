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
from app_funciones import *


##########################################################
##########################################################
##########################################################

def estrategias_page():
    st.header("Página: Estrategias")
    st.write("Aquí puedes documentar y explorar estrategias.")
    estrategia = st.text_area("Describe una estrategia:")
    if estrategia:
        st.write("Tu estrategia:")
        st.write(estrategia) 
    
    st.markdown("### Estrategia Global de Ventas")
    col1,col2,col3 = st.columns([1,4,1])
    with col2:
        st.image("temporadas.png")

    st.markdown(
        """ 
        #### Promociones Clave  
        Ajustamos las promociones según los patrones de compra y las asociaciones encontradas con Apriori:  
        - **Febrero (San Valentín):**  
          - Descuento en **Cream Hanging Heart T-Light Holder**.  
          - Pack promocional de **Regency Sugar Bowl Green y Regency Milk Jug Pink**, productos con fuerte vínculo de compra (Lift: 20.86).  
        - **Junio-Agosto (Verano):**  
          - Oferta en **Popcorn Holder**.  
          - Campañas de regreso a clases con descuentos en **Spaceboy Lunch Box y Dolly Girl Lunch Box**, que suelen comprarse juntas (Lift: 10.98).  
        - **Septiembre (Regreso a clases):**  
          - Enfoque en **World War 2 Gliders**.  
          - Venta cruzada entre **Lunch Bag Red Retrospot y Lunch Bag Black Skull** para incentivar compras combinadas (Lift: 6.30). 
          **Noviembre-Diciembre (Navidad):**  
          - Descuentos en **Paper Craft Little Birdie y T-Light Holder**.  
          - Pack especial en **Christmas Gingham Tree y Christmas Gingham Star**, productos clave para la temporada navideña (Lift: 20.48).  
          - Ofertas en **Paper Chain Kit Vintage Christmas y Paper Chain Kit 50’s Christmas**, artículos altamente comprados juntos (Lift: 10.66).  
        #### Gestión de Inventario  
        - Asegurar stock en temporadas clave, especialmente en productos con alta correlación de compra según Apriori.  
        - Evitar exceso de inventario en productos de baja rotación y aplicar descuentos agresivos en segmentos inactivos.  
        - Priorizar disponibilidad de productos estrella como **Regency Teacup Sets**, **Jumbo Bags** y **Alarm Clocks**, con alta compra conjunta.  
        """
    )
   
 

    st.markdown("---")
    st.markdown("### Estrategias por Segmento ")
    col1,col2,col3 = st.columns([1,4,1])
    with col2:
        st.image("segmentos.png")
    st.markdown(
        """
        #### **Segmento Excelente**  
        **Estrategias:**  
        - Descuentos progresivos y combos en **Regency Sugar Bowl Green y Regency Milk Jug Pink**, productos con fuerte vínculo de compra (Lift: 20.86).  
        - Ofertas en sets de decoración navideña con **Christmas Gingham Tree y Christmas Gingham Star** (Lift: 20.48).  
        - Packs de repostería con **Small Marshmallows Pink Bowl y Small Chocolates Pink Bowl**, que siempre se compran juntas (Confianza: 100%).  



        #### **Segmento Bueno**  
        **Estrategias:**  
        - Crear packs escolares con **Spaceboy Lunch Box y Dolly Girl Lunch Box** para la vuelta a clases.  
        - Ofrecer sets de vajilla vintage con **Green Regency Teacup and Saucer y Roses Regency Teacup and Saucer** (Lift: 19.81).  
        - Descuentos progresivos en **Jumbo Bag Red Retrospot y Jumbo Bag Pink Polkadot**, que suelen comprarse juntas.  


        #### **Segmento Regular**  
        **Estrategias:**  
        - Venta cruzada en campañas de decoración  con **Alarm Clock Bakelite Red y Alarm Clock Bakelite Green**.  
        - Promociones combinadas en **Lunch Bag Red Retrospot y Lunch Bag Black Skull** para incentivar compras escolares.  
        - Crear paquetes de decoración navideña con **Paper Chain Kit Vintage Christmas y Paper Chain Kit 50’s Christmas**.  



        #### **Segmento Inactivo**  
        **Estrategias:**  
        - Liquidación de stock con descuentos agresivos en **Jumbo Bag Red Retrospot y Jumbo Storage Bag Suki**.  
        - Campañas de email marketing para promocionar **Suki Shoulder Bag y Jumbo Bag Red Retrospot** (Lift: 6.24).  
        - Posicionamiento de **Picnic Basket Wicker Small y Jumbo Bag Red Retrospot** como productos ideales para eventos y viajes.  


        """
    )
    st.markdown("---")



    st.markdown("### Estrategias de Venta Cruzada (Apriori)")
    col1,col2,col3 = st.columns([1,4,1])
    with col2:
        st.image("cruzada.png")
    st.markdown(
        """
        #### Sugerencias en Carrito  
        - Si un cliente agrega **Lunch Bag Red Retrospot**, sugerirle **Lunch Bag Black Skull**.  
        - Si compra **Jumbo Bag Red Retrospot**, ofrecer un descuento en **Jumbo Bag Pink Polkadot**.  
        - Si compra **Paper Chain Kit Vintage Christmas**, sugerirle **Paper Chain Kit 50’s Christmas** con descuento.  

        #### Ubicación Estratégica en E-Commerce  
        - Mostrar productos relacionados en la misma sección para fomentar compras combinadas.  
        - Implementar recomendaciones automáticas de “Otros clientes también compraron”.  

        #### Campañas de Email Marketing Personalizado  
        - Si un cliente compra una **Regency Teacup**, enviarle promociones en otros modelos de la misma colección.  
        - Campañas de "Completa tu set" para clientes que compraron productos de colecciones.
        """
    )
    st.markdown("---")
    
    st.markdown(
        """
        ### Estrategia de Inventario Basado en Apriori  

        1. **Evitar quiebres de stock en productos comprados juntos.**  
           - Mantener suficiente inventario de productos con alta confianza de compra conjunta, como **Regency Sugar Bowl Green y Regency Milk Jug Pink**.  
           - Si se agota **Jumbo Bag Red Retrospot**, asegurar stock de **Jumbo Bag Pink Polkadot**.  

        2. **Gestión de stock en temporadas clave.**  
           - Tener suficiente inventario de **Lunch Bags** antes del regreso a clases.  
           - Garantizar stock de **kits de decoración navideña** desde octubre.  
        """
    )
    # with open("estrategias_final.md", "r") as file:
    #     contenido_markdown_estrategias = file.read()
    # st.markdown(contenido_markdown_estrategias)