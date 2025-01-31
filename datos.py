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


##########################################################
##########################################################
##########################################################

def datos_page():
    st.header("Página: Datos")
    st.write("Aquí puedes cargar y explorar los datos.")
    archivo = st.file_uploader("Carga un archivo CSV:", type=["csv"])
    if archivo:
        
        st.session_state.datos = pd.read_csv(archivo)
        st.write("Vista previa de los datos:")
        st.dataframe(st.session_state.datos)
        # Opciones de limpieza
        aplicar_limpieza = st.checkbox("Aplicar limpieza")
        if aplicar_limpieza and st.session_state.datos is not None:
            st.session_state.datos = limpieza(st.session_state.datos)
            st.write(f"Datos después de la limpieza, total de registros {len(st.session_state.datos)}")
            st.dataframe(st.session_state.datos)    