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

def resultados_page():
    st.header("Página: Resultados")
    st.write("Aquí puedes ver los resultados.")

    st.markdown(
        """
        # Clusterización usando Kmeans

        Usando los datos obtenidos de la segmentación, provenientes del modelo final (**Kmeans**), tenemos los promedios de las características por cluster.
        """)


    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        st.image("bar_graph.png")
    with col2:
        st.image("cluster.png")
    with col1:
        st.markdown(
            """
            | Métrica              | Valor                  |
            |----------------------|------------------------|
            | Silhouette Score    | 0.56711     |
            | Calinski-Harabasz   | 2833.86      |
            """
            )
    



    with open("/workspaces/Proyecto_dcd/resultados_app.md", "r") as file:
        contenido_markdown_resultados = file.read()
    st.markdown(contenido_markdown_resultados)
    