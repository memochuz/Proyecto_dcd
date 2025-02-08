import streamlit as st
from app_funciones import *
from intro import intro_page
from datos import datos_page
from analisis import analisis_page
from pca_kmeans import pca_kmeans_page
from market_basket import market_basket_page
from resultados import resultados_page
from estrategias import estrategias_page
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
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import matplotlib.ticker as ticker  


##########################################################
##########################################################
##########################################################

st.set_page_config(page_title="Dashboard de ML", layout="wide")


##########################################################
##########################################################
##########################################################

# Crear un estado inicial para los datos en session_state si no existe
if "datos" not in st.session_state:
    st.session_state.datos = None

if "rfm" not in st.session_state:
    st.session_state.rfm = None

if "rfm_std" not in st.session_state:
    st.session_state.rfm_std = None

if "y_hat_modelo_final" not in st.session_state:
    st.session_state.y_hat_modelo_final = None

if "score_modelo_final" not in st.session_state:
    st.session_state.score_modelo_final = None

if "modelo_final" not in st.session_state:
    st.session_state.modelo_final = None

if "columns_select_modelo_final" not in st.session_state:
    st.session_state.columns_select_modelo_final = None 

if "clusters_labels_final" not in st.session_state:
    st.session_state.clusters_labels_final = None

if "datos_etiquetados_final" not in st.session_state:
    st.session_state.datos_etiquetados_final = None
    
##########################################################
##########################################################
##########################################################


# Título principal
st.title("Proyecto: Segmentación y Análisis de Compras")

# Crear menú de navegación en la barra lateral
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Selecciona una página:",
    ["Introducción", "Datos", "Análisis", "PCA y Kmeans", "Market Basket","Resultados" ,"Estrategias"]
)

##########################################################
##########################################################
##########################################################

# Mostrar la página correspondiente

if pagina == "Introducción":
    intro_page()
elif pagina == "Datos":
    datos_page()
elif pagina == "Análisis":
    analisis_page()
elif pagina == "PCA y Kmeans":
    pca_kmeans_page()
elif pagina == "Market Basket":
    market_basket_page()
elif pagina == "Resultados":
    resultados_page()
elif pagina == "Estrategias":
    estrategias_page()

st.sidebar.info("Navega entre las diferentes secciones para explorar la aplicación.")
