import streamlit as st
from app_funciones import *
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

##########################################################
##########################################################
##########################################################

def market_basket_page():
    st.header("Análisis de Market Basket con Apriori")
    st.write("Aquí puedes entrenar y evaluar modelos de machine learning.")
    st.subheader("Algoritmo Apriori")

    st.write("Visualización de segmentos")
    ##########################################################
    ##########################################################
    ##########################################################

    # PROCESAMIENTO DE LOS DATOS

    # ADJUNTO LA CATEGORÍA A DATOS
    st.session_state.datos_etiquetados_final["Category"] = st.session_state.datos_etiquetados_final["Cluster"]\
                                                        .apply(lambda x : st.session_state.clusters_labels_final[x])

    # VISUALIZACIÓN SEGMENTOS
    segmento_seleccionado = st.selectbox(
        "Selecciona un segemento:",
        options=list(st.session_state.clusters_labels_final.values()),
    )

    # SELECCIONADO SEGMENTOS PARA UTILIZAR
    if segmento_seleccionado:
        st.write("Segmento basado en tú selección:")
        st.dataframe(
            st.session_state.datos_etiquetados_final[st.session_state.datos_etiquetados_final["Category"] == segmento_seleccionado]
        )
    else:
        st.warning("Por favor, selecciona al menos un segmento.") 


    segmentos_id = []
    for i in sorted(st.session_state.clusters_labels_final):
        segmentos_id.append(
            st.session_state.rfm[(st.session_state.rfm['Cluster'] == i)]["CustomerNo"])

    # LISTA CON LOS DATAFRAMES DE CADA SEGMENTO POR SEPARADO
    segmentos_df = []
    for segmento in segmentos_id:
        segmentos_df.append(
            st.session_state.datos_etiquetados_final[
                st.session_state.datos_etiquetados_final["CustomerNo"].isin(segmento)
            ]
        )

    # TRANSACCIONES POR SEGMENTO 
    transacciones_segmentos = []
    for df in segmentos_df:
        transacciones_segmentos.append(
            df.groupby(["TransactionNo","CustomerNo"])["ProductName"]\
            .apply(list).reset_index(drop=True)
        )
    
    # ONEHOTS POR TRANSACCIONES POR SEGMENTO
    onehots = []

    for i in range(len(transacciones_segmentos)):
      encoder = TransactionEncoder().fit(transacciones_segmentos[i])
      onehot = encoder.transform(transacciones_segmentos[i])
      onehot = pd.DataFrame(onehot, columns = encoder.columns_)
      onehots.append(onehot)

   # OBTENER LOS CONJUNTOS FRECUENTES CON EL ALGORITMO APRIORI CON UN DETERMINADO VALOR DE MIN_SUPPORT Y MAX_LEN
    freq_itemsets = []
    for i in range(len(onehots)):
      freq_itemsets.append(apriori(onehots[i], min_support=0.03, max_len = 3, use_colnames = True))

   # OBTENER LAS REGLAS DE ASOCIACIÓN
    rules = []
    for i in range(len(freq_itemsets)):
      rules.append(association_rules(freq_itemsets[i], metric="lift", min_threshold=1, num_itemsets=2)) 

    for i in range(len(rules)):
      rules[i]["lhs_items"] = rules[i].antecedents.apply(lambda x: len(x))
      rules[i]["rhs_items"] = rules[i].consequents.apply(lambda x: len(x))
      rules[i]["antecedents_"] = rules[i].antecedents.apply(lambda x: ",".join(list(x)))
      rules[i]["consequents_"] = rules[i].consequents.apply(lambda x: ",".join(list(x)))

    ##########################################################
    ##########################################################
    ##########################################################

    st.subheader("Reglas de asociación")

    # segmento_seleccionado_2 = st.selectbox(
    #    "Segemento:",
    #    options=["Excelente", "Bueno","Regular", "Inactivo"],
    # )

    if segmento_seleccionado:
        st.write("Métricas del respectivo segmento: ")
        diccionario_invertido = {v: k for k, v in st.session_state.clusters_labels_final.items()}
        st.dataframe(
            rules[diccionario_invertido[segmento_seleccionado]].loc[:,["antecedents_","consequents_","antecedent support","consequent support","support", "confidence","lift"]]
        )
    else:
        st.warning("Por favor, selecciona al menos un segmento.")
 
    ##########################################################
    ##########################################################
    ##########################################################

    # HEATMAPS
    st.subheader("Heatmaps")
    pivots_1_1 = []
    pivots_2_1 = []

    # TABLAS PIVOTE USANDO LIFT
    for i in range(len(rules)):
      pivots_1_1.append(rules[i][(rules[i].lhs_items == 1) & (rules[i].rhs_items == 1)].pivot(index = "antecedents_",columns = "consequents_",values = "lift"))
      pivots_2_1.append(rules[i][(rules[i].lhs_items == 2) & (rules[i].rhs_items == 1)].pivot(index = "antecedents_",columns = "consequents_",values = "lift"))

    # GRAFICAS POR SEGMENTO
    st.write("A continuación visualizamos las reglas de asociación que encontramos de 1 a 1 y 2 a 1:")

    col1, col2 = st.columns([1, 1])
    with col1:

        st.markdown("#### Reglas 1 a 1")

        segmento_seleccionado_2 = st.selectbox(
           "Heatmap (1 a 1) del segmento:",
           options=list(st.session_state.clusters_labels_final.values()),
        )

        if segmento_seleccionado_2 and not (pivots_1_1[diccionario_invertido[segmento_seleccionado_2]].empty):
            st.write("Heatmap basado en tú selección:")

            # SELECCIONO LAS REGLAS CON MAYOR LIFT QUE SON LAS QUE NOS INTERESAN
            
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(
               pivots_1_1[diccionario_invertido[segmento_seleccionado_2]], 
               annot=True, 
               ax=ax,
               annot_kws={"size": 9}
               )
            
            # Personalizar etiquetas
            ax.set_yticks(ax.get_yticks()) 
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)  # Tamaño de etiquetas en eje X
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)  # Tamaño de etiquetas en eje Y

            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.warning("Problema: selecciona al menos un segmento, o bien el segmento no es válido.") 


    with col2:
        st.markdown("#### Reglas 2 a 1")

        segmento_seleccionado_3 = st.selectbox(
           "Heatmap (2 a 1) del segmento:",
           options=list(st.session_state.clusters_labels_final.values()),
        )

        if segmento_seleccionado_3 and not (pivots_2_1[diccionario_invertido[segmento_seleccionado_3]].empty) :
            st.write("Heatmap basado en tú selección:")
            
            # SELECCIONO LAS REGLAS CON MAYOR LIFT QUE SON LAS QUE NOS INTERESAN

            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(pivots_2_1[diccionario_invertido[segmento_seleccionado_3]], annot=True, ax=ax)
            # Personalizar etiquetas
            ax.set_yticks(ax.get_yticks()) 
            ax.set_xticks(ax.get_xticks())
            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.warning("Problema: selecciona al menos un segmento, o bien el segmento no es válido.") 